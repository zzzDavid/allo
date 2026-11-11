# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool Gemini-II sessions at one revision under the recorded rule
(PAPER_REPRODUCTION.md, ISSUES X-34 / X-37): per row, the median of the
pooled Tenon samples (amortized device ticks per body, one per paired round)
over the candidate's validation session and its confirmation campaign;
REPRODUCED if the pooled median is at or below the paper's value with every
output exact, otherwise OPEN (nominal) when the paper's value lies inside the
pooled interquartile range and OPEN (regression) when it lies below it.

Every session is validated before anything is pooled (rounds 108-109, the
reviewer's incomplete-session, raw-disagreement, unpaired-rounds and
different-inputs probes): a session whose status is not ``pass`` or that
records problems is rejected; per case both lanes must measure exactly the
producer's rounds 1..N once each, every measured record must carry the
timing-evidence field explicitly and record no problem, ``outputs_exact``
recorded ``True`` (missing evidence is a rejection, never exactness), raw
repeated-batch device ticks that are positive and finite with the expected
launch count (one per launch of the selected plan on the Tenon lane, each
launch item's ``raw_ticks.pipeline`` equal to its batch record), and the
derived sample must equal ``sum(raw batch ticks) / repetitions`` as
``benchmarks/apu_g2/paper_rows.py`` derives it; the summarized lane samples
and median must equal the per-round values re-derived from the raw ticks;
the paired vendor/Tenon ratios must equal the ratios re-derived by matching
round identifiers; the two lanes must run the same fixture (identical
``inputs_sha256`` for the inputs and the oracle); the Tenon lane must carry
a program fingerprint, a layout fingerprint and a packing attestation on
every launch.  Across sessions the host, the installed package, the
protocol, the device lock, the paper values, the plan, the fixture hashes,
the program and layout fingerprints and the vendor baseline binaries must
agree.  The pooled samples are the values re-derived from the raw ticks.  Any rejection exits 2 and writes nothing unless
``--exclude-invalid`` is given, which pools only the valid sessions and
lists the rejected ones in the output.

    python g2_pool.py --out pooled.json session.json [campaign_s1.json ...]
"""

import argparse
import json
import math
import statistics
import sys

TENON_LAUNCH_FIELDS = (
    "program_fingerprint",
    "layout_fingerprint",
    "packing_attestation",
)


class SessionInvalid(Exception):
    pass


def quartiles(xs):
    xs = sorted(xs)
    n = len(xs)
    return {
        "samples": n,
        "median": statistics.median(xs),
        "q1": statistics.median(xs[: n // 2]) if n > 1 else xs[0],
        "q3": statistics.median(xs[(n + 1) // 2 :]) if n > 1 else xs[0],
        "min": xs[0],
        "max": xs[-1],
    }


def _require(condition, message):
    if not condition:
        raise SessionInvalid(message)


def _positive_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _close(a, b):
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


def _fixture_identity(name, lane, inputs):
    """The fixture a lane ran: the sha256 of every input and of the oracle."""
    _require(
        isinstance(inputs, dict) and inputs, f"{name}/{lane}: no inputs_sha256 recorded"
    )
    _require("oracle" in inputs, f"{name}/{lane}: inputs_sha256 records no oracle")
    for key, value in inputs.items():
        _require(
            isinstance(value, str) and len(value) >= 16,
            f"{name}/{lane}: inputs_sha256[{key!r}] is not a digest",
        )
    return tuple(sorted(inputs.items()))


def validate_session(d):
    """Return the per-case evidence of one session or raise SessionInvalid."""
    _require(d.get("status") == "pass", f"status is {d.get('status')!r}, not 'pass'")
    _require(d.get("problems") == [], f"problems recorded: {d.get('problems')!r}")
    for key in ("host", "allo", "protocol", "device_lock", "cases", "launches"):
        _require(key in d, f"missing field {key!r}")
    protocol = d["protocol"]
    rounds = protocol.get("paired_rounds")
    _require(
        isinstance(rounds, int) and not isinstance(rounds, bool) and rounds > 0,
        f"protocol.paired_rounds is {rounds!r}",
    )
    repetitions = protocol.get("repetitions")
    _require(
        isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and repetitions > 0,
        f"protocol.repetitions is {repetitions!r}",
    )
    _require(isinstance(d["launches"], list) and d["launches"], "no launch records")
    cases = {}
    for name, c in d["cases"].items():
        lanes = c.get("lanes") or {}
        _require("tenon" in lanes, f"{name}: no tenon lane")
        vendor_key = (
            "vendor" if "vendor" in lanes else ("gtml" if "gtml" in lanes else None)
        )
        _require(vendor_key is not None, f"{name}: no vendor lane")
        _require(
            isinstance(c.get("paper_tenon"), (int, float)), f"{name}: no paper_tenon"
        )
        inputs = _fixture_identity(name, "tenon", lanes["tenon"].get("inputs_sha256"))
        _require(
            inputs
            == _fixture_identity(
                name, "vendor", lanes[vendor_key].get("inputs_sha256")
            ),
            f"{name}: the tenon and vendor lanes record different fixtures (inputs_sha256)",
        )
        evidence = {
            "case_id": c.get("case_id"),
            "paper_tenon": c["paper_tenon"],
            "paper_gtml": c.get("paper_gtml"),
            "plan": lanes["tenon"].get("plan"),
            "inputs": inputs,
            "program_fingerprints": set(),
            "layout_fingerprints": set(),
            "vendor_binaries": set(),
            "samples": {},
            "by_round": {},
            "paired": [],
        }
        for lane_name, lane_key in (("tenon", "tenon"), ("vendor", vendor_key)):
            measured = [
                l
                for l in d["launches"]
                if l.get("case") in (name, c.get("case_id"))
                and l.get("lane") == lane_name
                and l.get("kind") == "measured"
            ]
            _require(
                len(measured) == rounds,
                f"{name}/{lane_name}: {len(measured)} measured launches, {rounds} paired rounds",
            )
            seen = sorted(l.get("round") for l in measured)
            # paper_rows numbers the warmup round 0 and the measured rounds 1..N;
            # both lanes must measure exactly those rounds, once each.
            _require(
                seen == list(range(1, rounds + 1)),
                f"{name}/{lane_name}: measured rounds {seen}, rounds 1..{rounds} required",
            )
            by_round = {}
            for l in measured:
                where = f"{name}/{lane_name} round {l.get('round')}"
                _require(
                    "timing_evidence_problem" in l,
                    f"{where}: timing-evidence field absent",
                )
                _require(
                    l["timing_evidence_problem"] is None,
                    f"{where}: timing evidence problem {l['timing_evidence_problem']!r}",
                )
                _require(
                    l.get("outputs_exact") is True,
                    f"{where}: outputs_exact is {l.get('outputs_exact')!r}",
                )
                batch = l.get("repeated_batch_ticks")
                _require(
                    isinstance(batch, list) and batch,
                    f"{where}: no raw repeated-batch ticks",
                )
                for index, ticks in enumerate(batch):
                    _require(
                        _positive_finite(ticks),
                        f"{where}: launch {index} raw device ticks {ticks!r} are not a positive finite count",
                    )
                if lane_name == "tenon":
                    items = l.get("launches")
                    _require(
                        isinstance(items, list) and items, f"{where}: no launch items"
                    )
                    expected = l.get("expected_launches")
                    _require(
                        isinstance(expected, int)
                        and expected > 0
                        and expected == len(items) == len(batch),
                        f"{where}: {len(items)} launch items and {len(batch)} batch records for a plan of {expected!r} launch(es)",
                    )
                    for index, item in enumerate(items):
                        for field in TENON_LAUNCH_FIELDS:
                            _require(
                                item.get(field),
                                f"{where}: launch {index} without {field}",
                            )
                        pipeline = (item.get("raw_ticks") or {}).get("pipeline")
                        _require(
                            _positive_finite(pipeline) and pipeline == batch[index],
                            f"{where}: launch {index} raw_ticks.pipeline {pipeline!r} does not match its batch record {batch[index]!r}",
                        )
                        evidence["program_fingerprints"].add(
                            item["program_fingerprint"]
                        )
                        evidence["layout_fingerprints"].add(item["layout_fingerprint"])
                else:
                    _require(
                        l.get("host_sha256") and l.get("device_library_sha256"),
                        f"{where}: baseline binaries not recorded",
                    )
                    evidence["vendor_binaries"].add(
                        (l["host_sha256"], l["device_library_sha256"])
                    )
                derived = sum(float(t) for t in batch) / repetitions
                recorded = l.get("amortized_ticks_per_pipeline")
                _require(
                    _positive_finite(recorded) and _close(recorded, derived),
                    f"{where}: derived sample {recorded!r} disagrees with the raw ticks ({batch} / {repetitions} = {derived})",
                )
                by_round[l["round"]] = derived
            per_round = [by_round[r] for r in range(1, rounds + 1)]
            summarized = lanes[lane_key].get("amortized_ticks")
            _require(
                isinstance(summarized, list)
                and len(summarized) == rounds
                and all(
                    _close(a, b) for a, b in zip(sorted(summarized), sorted(per_round))
                ),
                f"{name}/{lane_name}: summarized samples do not equal the per-round values re-derived from the raw ticks",
            )
            stats = lanes[lane_key].get("statistics") or {}
            if "median" in stats:
                _require(
                    _close(stats["median"], statistics.median(per_round)),
                    f"{name}/{lane_name}: summarized median differs from the raw-derived samples",
                )
            evidence["samples"][lane_name] = per_round
            evidence["by_round"][lane_name] = by_round
        ratios = [
            evidence["by_round"]["vendor"][r] / evidence["by_round"]["tenon"][r]
            for r in range(1, rounds + 1)
        ]
        recorded_ratios = c.get("comparison", {}).get(
            "paired_round_ratios_vendor_over_tenon"
        )
        _require(
            isinstance(recorded_ratios, list)
            and len(recorded_ratios) == rounds
            and all(
                _close(a, b) for a, b in zip(sorted(recorded_ratios), sorted(ratios))
            ),
            f"{name}: the recorded paired vendor/tenon ratios do not equal the ratios re-derived by round",
        )
        ratio_stats = c.get("comparison", {}).get("paired_ratio_statistics") or {}
        if "median" in ratio_stats:
            _require(
                _close(ratio_stats["median"], statistics.median(ratios)),
                f"{name}: the paired-ratio median differs from the ratios re-derived by round",
            )
        evidence["paired"] = ratios
        _require(
            len(evidence["program_fingerprints"]) == 1,
            f"{name}: {len(evidence['program_fingerprints'])} program fingerprints in one session",
        )
        _require(
            len(evidence["layout_fingerprints"]) == 1,
            f"{name}: {len(evidence['layout_fingerprints'])} layout fingerprints in one session",
        )
        _require(
            len(evidence["vendor_binaries"]) == 1,
            f"{name}: {len(evidence['vendor_binaries'])} vendor baseline binaries in one session",
        )
        cases[name] = evidence
    _require(cases, "no cases")
    return {
        "host": d["host"],
        "allo": d["allo"],
        "protocol": protocol,
        "device_lock": d["device_lock"],
        "cases": cases,
    }


def check_compatible(reference, other, label):
    for key in ("host", "allo", "protocol", "device_lock"):
        _require(
            reference[key] == other[key],
            f"{label}: {key} differs from the first session ({other[key]!r} vs {reference[key]!r})",
        )
    _require(
        set(reference["cases"]) == set(other["cases"]), f"{label}: case set differs"
    )
    for name, a in reference["cases"].items():
        b = other["cases"][name]
        for key in (
            "case_id",
            "paper_tenon",
            "paper_gtml",
            "plan",
            "inputs",
            "program_fingerprints",
            "layout_fingerprints",
            "vendor_binaries",
        ):
            _require(
                a[key] == b[key],
                f"{label}: {name} {key} differs from the first session",
            )


def pool(validated):
    rows = {}
    reference = validated[0]
    for name, ref in reference["cases"].items():
        tenon, gtml, paired, session_medians = [], [], [], []
        for v in validated:
            e = v["cases"][name]
            tenon += e["samples"]["tenon"]
            gtml += e["samples"]["vendor"]
            paired += e["paired"]
            session_medians.append(statistics.median(e["samples"]["tenon"]))
        t = quartiles(tenon)
        paper = ref["paper_tenon"]
        if t["median"] <= paper:
            status = "REPRODUCED"  # every launch's outputs_exact was True in validation
        elif t["q1"] <= paper <= t["q3"]:
            status = "OPEN (nominal)"
        else:
            status = "OPEN (regression)"
        rows[name] = {
            "case_id": ref["case_id"],
            "paper_tenon": paper,
            "pooled_tenon": t,
            "ratio_to_paper": t["median"] / paper,
            "session_medians": session_medians,
            "gtml_pooled": quartiles(gtml),
            "paper_gtml": ref["paper_gtml"],
            "paired_vendor_over_tenon_median": (
                statistics.median(paired) if paired else None
            ),
            "paper_gtml_over_tenon": (
                (ref["paper_gtml"] / paper) if ref["paper_gtml"] else None
            ),
            "plan": ref["plan"],
            "inputs_sha256": dict(ref["inputs"]),
            "program_fingerprint": sorted(ref["program_fingerprints"])[0],
            "layout_fingerprint": sorted(ref["layout_fingerprints"])[0],
            "outputs_exact": True,
            "status": status,
        }
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--exclude-invalid",
        action="store_true",
        help="pool the valid sessions and list the rejected ones instead of failing",
    )
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    validated, accepted, rejected = [], [], []
    for f in args.files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            v = validate_session(d)
            if validated:
                check_compatible(validated[0], v, f)
            validated.append(v)
            accepted.append(
                {
                    "file": f,
                    "host": d["host"],
                    "date": d.get("date"),
                    "allo": d["allo"],
                    "launches": len(d["launches"]),
                }
            )
        except (SessionInvalid, OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({"file": f, "reason": str(exc)})
            print(f"REJECTED {f}: {exc}", file=sys.stderr)
    if rejected and not args.exclude_invalid:
        print(f"{len(rejected)} session(s) rejected; nothing pooled", file=sys.stderr)
        return 2
    if not validated:
        print("no valid session", file=sys.stderr)
        return 2
    rows = pool(validated)
    for name, r in rows.items():
        t = r["pooled_tenon"]
        print(
            f"{name}: pooled median {t['median']} / paper {r['paper_tenon']} ratio {r['ratio_to_paper']:.4f} IQR [{t['q1']}, {t['q3']}] n={t['samples']} sessions {r['session_medians']} exact={r['outputs_exact']} -> {r['status']}"
        )
    with open(args.out, "w") as fh:
        json.dump(
            {"sessions": accepted, "rejected_sessions": rejected, "rows": rows},
            fh,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
