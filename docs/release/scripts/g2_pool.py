"""Pool Gemini-II sessions at one revision under the recorded rule
(PAPER_REPRODUCTION.md, ISSUES X-34 / X-37): per row, the median of the
pooled Tenon samples (amortized device ticks per body, one per paired round)
over the candidate's validation session and its confirmation campaign;
REPRODUCED if the pooled median is at or below the paper's value with every
output exact, otherwise OPEN (nominal) when the paper's value lies inside the
pooled interquartile range and OPEN (regression) when it lies below it.

Every session is validated before anything is pooled (round 108, the
reviewer's incomplete-session probe): a session whose status is not ``pass``
or that records problems is rejected; per case and lane the measured
launches must cover every paired round exactly once with ``outputs_exact``
recorded ``True`` (missing evidence is a rejection, never exactness), no
timing-evidence problem, the expected launch count, and the summarized
``amortized_ticks`` must equal the measured launches' per-round values; the
Tenon lane must carry a program fingerprint, a layout fingerprint and a
packing attestation on every launch.  Across sessions the host, the
installed package, the protocol, the device lock, the paper values, the
plan, the program and layout fingerprints and the vendor baseline binaries
must agree.  Any rejection exits 2 and writes nothing unless
``--exclude-invalid`` is given, which pools only the valid sessions and
lists the rejected ones in the output.

    python g2_pool.py --out pooled.json session.json [campaign_s1.json ...]
"""

import argparse
import json
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


def validate_session(d):
    """Return the per-case evidence of one session or raise SessionInvalid."""
    _require(d.get("status") == "pass", f"status is {d.get('status')!r}, not 'pass'")
    _require(d.get("problems") == [], f"problems recorded: {d.get('problems')!r}")
    for key in ("host", "allo", "protocol", "device_lock", "cases", "launches"):
        _require(key in d, f"missing field {key!r}")
    protocol = d["protocol"]
    rounds = protocol.get("paired_rounds")
    _require(
        isinstance(rounds, int) and rounds > 0, f"protocol.paired_rounds is {rounds!r}"
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
        evidence = {
            "case_id": c.get("case_id"),
            "paper_tenon": c["paper_tenon"],
            "paper_gtml": c.get("paper_gtml"),
            "plan": lanes["tenon"].get("plan"),
            "program_fingerprints": set(),
            "layout_fingerprints": set(),
            "vendor_binaries": set(),
            "samples": {},
            "paired": list(
                c.get("comparison", {}).get("paired_round_ratios_vendor_over_tenon", [])
            ),
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
            # each measured round exactly once, consecutively.
            _require(
                all(isinstance(r, int) for r in seen)
                and seen == list(range(seen[0], seen[0] + rounds)),
                f"{name}/{lane_name}: measured rounds covered {seen}, {rounds} consecutive rounds required",
            )
            per_round = []
            for l in measured:
                _require(
                    l.get("outputs_exact") is True,
                    f"{name}/{lane_name} round {l.get('round')}: outputs_exact is {l.get('outputs_exact')!r}",
                )
                _require(
                    l.get("timing_evidence_problem") is None,
                    f"{name}/{lane_name} round {l.get('round')}: timing evidence problem {l.get('timing_evidence_problem')!r}",
                )
                ticks = l.get("amortized_ticks_per_pipeline")
                _require(
                    isinstance(ticks, (int, float)) and ticks > 0,
                    f"{name}/{lane_name} round {l.get('round')}: no amortized ticks",
                )
                per_round.append(ticks)
                if lane_name == "tenon":
                    items = l.get("launches")
                    _require(
                        isinstance(items, list) and items,
                        f"{name}/tenon round {l.get('round')}: no launch items",
                    )
                    _require(
                        l.get("expected_launches") == len(items),
                        f"{name}/tenon round {l.get('round')}: {len(items)} launch items, expected {l.get('expected_launches')!r}",
                    )
                    for item in items:
                        for field in TENON_LAUNCH_FIELDS:
                            _require(
                                item.get(field),
                                f"{name}/tenon round {l.get('round')}: launch without {field}",
                            )
                        evidence["program_fingerprints"].add(
                            item["program_fingerprint"]
                        )
                        evidence["layout_fingerprints"].add(item["layout_fingerprint"])
                else:
                    _require(
                        l.get("host_sha256") and l.get("device_library_sha256"),
                        f"{name}/vendor round {l.get('round')}: baseline binaries not recorded",
                    )
                    evidence["vendor_binaries"].add(
                        (l["host_sha256"], l["device_library_sha256"])
                    )
            summarized = lanes[lane_key].get("amortized_ticks")
            _require(
                isinstance(summarized, list)
                and sorted(summarized) == sorted(per_round),
                f"{name}/{lane_name}: summarized samples do not equal the measured launches' per-round ticks",
            )
            stats = lanes[lane_key].get("statistics") or {}
            if "median" in stats:
                _require(
                    abs(stats["median"] - statistics.median(per_round)) < 1e-9,
                    f"{name}/{lane_name}: summarized median differs from the launches",
                )
            evidence["samples"][lane_name] = per_round
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
