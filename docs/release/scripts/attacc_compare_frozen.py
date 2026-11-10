"""Compare an installed-artifact AttAcc session set with the frozen paper's figures
(docs/release/paper_reproduction/inventory.csv rows attacc_kernel_* and attacc_e2e_*).

    python attacc_compare_frozen.py --sessions-dir DIR --inventory inventory.csv --manifest tenon_manifest.csv --out report.json

Kernel rows qkv_b1, qkv_b4, proj_b1, proj_b4: the shared-pipeline selection's Ramulator2-AttAcc
milliseconds against the paper's value, and the re-run fixed-form baseline.  Score rows: the mean
over 248 shapes of fixed / Tenon (the figure's statistic) with the fixed per-shape times from the
archived manifest, and the per-shape tally against DCC's archived times (ties at equal
milliseconds, as the round 51 tally).  Decode configurations: base and full speedups from
compose_e2e against the paper's.  Every run must be `simulator-timing-only` with a matching
trace check; any failure or missing row is reported and the exit status is nonzero.
"""

import argparse, csv, json, statistics, sys
from pathlib import Path

ROWS = {
    "qkv_b1": "attacc_kernel_qkv_b1",
    "qkv_b4": "attacc_kernel_qkv_b4",
    "proj_b1": "attacc_kernel_proj_b1",
    "proj_b4": "attacc_kernel_proj_b4",
}
SCORE = {
    "score_b1": ("attacc_kernel_score_b1", "8"),
    "score_b4": ("attacc_kernel_score_b4", "32"),
}
E2E = {
    (128, 2048, 1): "attacc_e2e_in128_out2048_bs1",
    (2048, 128, 1): "attacc_e2e_in2048_out128_bs1",
    (2048, 2048, 1): "attacc_e2e_in2048_out2048_bs1",
    (128, 2048, 4): "attacc_e2e_in128_out2048_bs4",
    (2048, 128, 4): "attacc_e2e_in2048_out128_bs4",
    (2048, 2048, 4): "attacc_e2e_in2048_out2048_bs4",
}


def paper_values(inventory):
    rows = {r["case_id"]: r for r in csv.DictReader(open(inventory))}
    return rows


def check_runs(session, problems, label):
    if session.get("failures"):
        problems.append(f"{label}: session records failures: {session['failures']}")
    for run in session["runs"]:
        if run.get("kind") != "simulator-timing-only":
            problems.append(f"{label}: run kind {run.get('kind')!r}")
        if run.get("selection") == "shared-pipeline":
            # the compiler's program is checked command by command; the fixed-form
            # baseline is the paper's reference tiling, not a compiled program
            tc = run.get("trace_check") or {}
            if tc.get("match") is not True or tc.get("totals_match") is not True:
                problems.append(
                    f"{label}: trace check did not match for shape {run.get('shape')}"
                )
        if (
            not isinstance(run.get("milliseconds"), (int, float))
            or run["milliseconds"] <= 0
        ):
            problems.append(
                f"{label}: no positive milliseconds for shape {run.get('shape')}"
            )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument(
        "--manifest",
        required=True,
        help="archived tenon_manifest.csv (fixed and DCC per-shape ms)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    S = Path(args.sessions_dir)
    paper = paper_values(args.inventory)
    manifest = {
        (r["nhead"], r["input_dim"], r["output_dim"]): r
        for r in csv.DictReader(open(args.manifest))
    }
    problems, report = [], {
        "kernel_rows": {},
        "score_rows": {},
        "decode": {},
        "simulator": None,
    }
    gemv_tally = []
    for row, case in ROWS.items():
        f = S / f"attacc_{row}.json"
        if not f.exists():
            problems.append(f"{row}: session missing")
            continue
        d = json.load(open(f))
        check_runs(d, problems, row)
        report["simulator"] = report["simulator"] or {
            "name": d["simulator"].get("name"),
            "binary_sha256": d["simulator"].get("binary_sha256"),
            "trace_generator": d.get("trace_generator", {}).get("files_sha256"),
        }
        sel = [r for r in d["runs"] if r["selection"] == "shared-pipeline"]
        fixed = [r for r in d["runs"] if r["selection"] == "fixed-form-baseline"]
        if len(sel) != 1 or not fixed:
            problems.append(
                f"{row}: expected one selected run and fixed-form baseline runs, got {len(sel)} / {len(fixed)}"
            )
            continue
        ms = sel[0]["milliseconds"]
        fixed_ms = min(r["milliseconds"] for r in fixed)
        h, dim, l = sel[0]["shape"]
        m = manifest.get((str(h), str(dim), str(l)))
        if m is None:
            problems.append(
                f"{row}: shape {sel[0]['shape']} not in the archived manifest"
            )
        else:
            gemv_tally.append((ms, float(m["dcc_our_time_ms"])))
        pv = float(paper[case]["paper_value"])
        ratio = ms / pv
        verdict = "REPRODUCED" if abs(ratio - 1) <= 0.0015 else "DIFFERS"
        report["kernel_rows"][row] = {
            "case_id": case,
            "selected_ms": ms,
            "paper_ms": pv,
            "ratio": ratio,
            "fixed_baseline_ms": fixed_ms,
            "speedup_over_fixed": fixed_ms / ms,
            "tiling": sel[0].get("tiling"),
            "verdict": verdict,
        }
        print(
            f"{row}: {ms:.6f} ms vs paper {pv} (ratio {ratio:.4f}); fixed {fixed_ms:.6f}; speedup {fixed_ms/ms:.2f}x -> {verdict}"
        )
    tally = {"wins": 0, "ties": 0, "losses": 0, "loss_pct": []}
    for t, dcc_ms in gemv_tally:  # the text's tally covers all 500 kernels
        if abs(t - dcc_ms) <= 1e-12:
            tally["ties"] += 1
        elif t < dcc_ms:
            tally["wins"] += 1
        else:
            tally["losses"] += 1
            tally["loss_pct"].append((t / dcc_ms - 1) * 100)
    for row, (case, nhead) in SCORE.items():
        f = S / f"attacc_{row}.json"
        if not f.exists():
            problems.append(f"{row}: session missing")
            continue
        d = json.load(open(f))
        check_runs(d, problems, row)
        ratios, dcc = [], []
        for run in d["runs"]:
            h, dim, l = run["shape"]
            m = manifest.get((str(h), str(dim), str(l)))
            if m is None:
                problems.append(
                    f"{row}: shape {run['shape']} not in the archived manifest"
                )
                continue
            fixed_ms, dcc_ms, t = (
                float(m["fixed_baseline_ms"]),
                float(m["dcc_our_time_ms"]),
                run["milliseconds"],
            )
            ratios.append(fixed_ms / t)
            if abs(t - dcc_ms) <= 1e-12:
                tally["ties"] += 1
            elif t < dcc_ms:
                tally["wins"] += 1
            else:
                tally["losses"] += 1
                tally["loss_pct"].append((t / dcc_ms - 1) * 100)
        figure = paper[case]["paper_value"]
        mean_ratio = statistics.mean(ratios) if ratios else float("nan")
        target = float(figure.split("x")[0].split()[-1])
        verdict = (
            "REPRODUCED" if ratios and round(mean_ratio, 2) == target else "DIFFERS"
        )
        report["score_rows"][row] = {
            "case_id": case,
            "shapes": len(d["runs"]),
            "mean_ms": statistics.mean(r["milliseconds"] for r in d["runs"]),
            "speedup_over_fixed_mean_of_ratios": mean_ratio,
            "figure": figure,
            "verdict": verdict,
        }
        print(
            f"{row}: {len(d['runs'])} shapes, speedup over fixed {mean_ratio:.3f}x vs the figure's {target}x -> {verdict}"
        )
    tally["median_loss_pct"] = (
        statistics.median(tally["loss_pct"]) if tally["loss_pct"] else None
    )
    tally.pop("loss_pct")
    text_tally = {"wins": 123, "ties": 316, "losses": 61}
    tally["matches_text"] = (
        all(tally[k] == v for k, v in text_tally.items())
        and round(tally["median_loss_pct"] or 0, 1) == 7.1
    )
    report["score_rows"]["tally_vs_dcc_all_500"] = tally
    print(
        "tally vs DCC over all 500 kernels (text: 123 / 316 / 61, median loss 7.1%):",
        tally,
    )
    if not tally["matches_text"]:
        problems.append("the 500-instance tally against DCC differs from the text's")
    f = S / "attacc_e2e.json"
    if not f.exists():
        problems.append("e2e: composition missing")
    else:
        e = json.load(open(f))
        for cfg in e["configs"]:
            key = (cfg["lin"], cfg["lout"], cfg["batch"])
            case = E2E.get(key)
            if not case:
                continue
            t = cfg["variants"]["tenon"]
            base, full = t["base_speedup"], t["full_speedup"]
            pv = paper[case]["paper_value"]  # "base 1.47x, full 6.5x"
            pb = float(pv.split("base")[1].split("x")[0])
            pf = float(pv.split("full")[1].split("x")[0])
            verdict = (
                "REPRODUCED"
                if round(base, 2) == round(pb, 2) and round(full, 2) == round(pf, 2)
                else "DIFFERS"
            )
            report["decode"][case] = {
                "base_speedup": base,
                "full_speedup": full,
                "paper": pv,
                "verdict": verdict,
                "dcc_full": cfg["variants"].get("dcc", {}).get("full_speedup"),
                "fixed_full": cfg["variants"].get("fixed", {}).get("full_speedup"),
            }
            print(
                f"{case}: base {base:.2f}x full {full:.2f}x vs paper {pv} -> {verdict}"
            )
    report["problems"] = problems
    verdicts = [
        v["verdict"]
        for sec in ("kernel_rows", "score_rows", "decode")
        for k, v in report[sec].items()
        if isinstance(v, dict) and "verdict" in v
    ]
    report["all_reproduced"] = (
        bool(verdicts) and all(v == "REPRODUCED" for v in verdicts) and not problems
    )
    json.dump(report, open(args.out, "w"), indent=1)
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    print("ALL REPRODUCED" if report["all_reproduced"] else "NOT ALL REPRODUCED")
    return 0 if report["all_reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
