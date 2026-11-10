"""Pool Gemini-II sessions at one revision under the recorded rule
(PAPER_REPRODUCTION.md, ISSUES X-34 / X-37): per row, the median of the
pooled Tenon samples (amortized device ticks per body, one per paired round)
over the candidate's validation session and its confirmation campaign;
REPRODUCED if the pooled median is at or below the paper's value with every
output exact, otherwise OPEN (nominal) when the paper's value lies inside the
pooled interquartile range and OPEN (regression) when it lies below it.  Also
reported: per-session medians, the pooled GTML lane, the paired GTML/Tenon
ratio against the paper's, fingerprints and the session statuses.

    python g2_pool.py --out pooled.json session.json [campaign_s1.json ...]
"""
import argparse
import json
import statistics


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    pooled, sessions = {}, []
    for f in args.files:
        d = json.load(open(f))
        exact = {}
        for launch in d.get("launches", []):
            exact.setdefault(launch["case"], True)
            if launch.get("outputs_exact") is not True:
                exact[launch["case"]] = False
        sessions.append({"file": f, "host": d.get("host"), "date": d.get("date"), "allo": d.get("allo"), "status": d.get("status"), "problems": d.get("problems"), "launches": len(d.get("launches", []))})
        for name, c in d["cases"].items():
            tenon = c["lanes"]["tenon"]
            vendor = c["lanes"].get("vendor") or c["lanes"].get("gtml") or {}
            e = pooled.setdefault(name, {"case_id": c.get("case_id"), "paper_tenon": c["paper_tenon"], "paper_gtml": c.get("paper_gtml"), "tenon": [], "gtml": [], "session_medians": [], "paired_vendor_over_tenon": [], "plans": set(), "program_fingerprints": set(), "outputs_exact": True})
            e["tenon"] += tenon["amortized_ticks"]
            e["session_medians"].append(statistics.median(tenon["amortized_ticks"]))
            e["gtml"] += vendor.get("amortized_ticks", [])
            e["paired_vendor_over_tenon"] += c.get("comparison", {}).get("paired_round_ratios_vendor_over_tenon", [])
            e["plans"].add(tenon.get("plan"))
            e["outputs_exact"] = e["outputs_exact"] and exact.get(name, exact.get(c.get("case_id"), True))
            for launch in d.get("launches", []):
                if launch["case"] in (name, c.get("case_id")) and launch["lane"] == "tenon":
                    for item in launch.get("launches", []):
                        e["program_fingerprints"].add(item.get("program_fingerprint"))
    out = {"sessions": sessions, "rows": {}}
    for name, e in pooled.items():
        t = quartiles(e["tenon"])
        paper = e["paper_tenon"]
        if t["median"] <= paper and e["outputs_exact"]:
            status = "REPRODUCED"
        elif t["q1"] <= paper <= t["q3"]:
            status = "OPEN (nominal)"
        elif paper < t["q1"]:
            status = "OPEN (regression)"
        else:
            status = "OPEN (nominal; the paper's value above the pooled IQR)" if not e["outputs_exact"] else "REPRODUCED"
        out["rows"][name] = {
            "case_id": e["case_id"], "paper_tenon": paper, "pooled_tenon": t, "ratio_to_paper": t["median"] / paper,
            "session_medians": e["session_medians"], "gtml_pooled": quartiles(e["gtml"]) if e["gtml"] else None,
            "paper_gtml": e["paper_gtml"], "paired_vendor_over_tenon_median": statistics.median(e["paired_vendor_over_tenon"]) if e["paired_vendor_over_tenon"] else None,
            "paper_gtml_over_tenon": (e["paper_gtml"] / paper) if e["paper_gtml"] else None,
            "plans": sorted(p for p in e["plans"] if p), "program_fingerprints": sorted(p for p in e["program_fingerprints"] if p),
            "outputs_exact": e["outputs_exact"], "status": status,
        }
        r = out["rows"][name]
        print(f"{name}: pooled median {t['median']} / paper {paper} ratio {r['ratio_to_paper']:.4f} IQR [{t['q1']}, {t['q3']}] n={t['samples']} sessions {e['session_medians']} exact={e['outputs_exact']} -> {status}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
