"""Compare an installed-artifact AttAcc session set with the frozen paper's figures.

    python attacc_compare_frozen.py --sessions-dir DIR --inventory inventory.csv \\
        --manifest tenon_manifest.csv --out report.json

Evidence contract (round 115, after the review of round 114 showed the first
version certified incomplete or relabelled evidence):

* GEMV rows qkv_b1, qkv_b4, proj_b1, proj_b4: exactly one `shared-pipeline`
  run of the row's expected shape (from the archived manifest), at least one
  `fixed-form-baseline` run, every run `simulator-timing-only`, the selected
  run's trace check matched with totals.
* Score rows score_b1, score_b4: exactly the archived manifest's attention
  shapes of the row's head count, each once, every record `shared-pipeline`
  with a matching trace check; records of another selection (reference or
  forced tilings), another row's shapes, duplicates or missing shapes are
  rejected by name.
* Decode: the composition record of kind `analytical-composition` with exactly
  the six expected (lin, lout, batch) configurations, no duplicates.
* Performance rule, same-or-better: a selected timing REPRODUCES when it is at
  or below the archived per-shape Tenon timing of the frozen experiment
  (`tenon_time_ms` in the manifest, deterministic); where no archived timing
  exists the printed paper value is used with a half-unit-in-the-last-place
  allowance derived from the digits printed, in the favourable direction only.
  Score-row speedups and decode speedups REPRODUCE when at or above the printed
  figure less half a unit of its last printed digit.  Improvements pass;
  regressions fail.
* The 500-instance tally against DCC (123 / 316 / 61, median loss 7.1 percent)
  is reported as a diagnostic and does not decide the verdict.
* Success requires all twelve result verdicts present and REPRODUCED and no
  evidence problem; otherwise the exit status is 1.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROWS = {
    "qkv_b1": ("attacc_kernel_qkv_b1", "8", "5120", "384"),
    "qkv_b4": ("attacc_kernel_qkv_b4", "32", "5120", "384"),
    "proj_b1": ("attacc_kernel_proj_b1", "8", "128", "5120"),
    "proj_b4": ("attacc_kernel_proj_b4", "32", "128", "5120"),
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
TEXT_TALLY = {"wins": 123, "ties": 316, "losses": 61, "median_loss_pct": 7.1}
EPS = 1e-12


def half_ulp(printed: str) -> float:
    """Half a unit of the last printed digit of a decimal literal ("0.00881" -> 5e-6)."""
    text = printed.strip()
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** (-decimals)


def first_number(text: str, after: str = "") -> str:
    """The first decimal literal in `text` following `after` (as printed)."""
    tail = text.split(after, 1)[1] if after else text
    match = re.search(r"\d+(?:\.\d+)?", tail)
    if not match:
        raise ValueError(f"no number in {text!r}")
    return match.group(0)


def load_manifest(path):
    rows = list(csv.DictReader(open(path)))
    by_shape = {(r["nhead"], r["input_dim"], r["output_dim"]): r for r in rows}
    attention = {
        nhead: {
            (r["nhead"], r["input_dim"], r["output_dim"])
            for r in rows
            if r["family"] != "GEMV" and r["nhead"] == nhead
        }
        for nhead in {r["nhead"] for r in rows}
    }
    return by_shape, attention


def shape_key(run):
    try:
        h, dim, l = run["shape"]
    except (KeyError, TypeError, ValueError):
        return None
    return (str(h), str(dim), str(l))


def check_run(run, label, problems, *, require_trace):
    if run.get("kind") != "simulator-timing-only":
        problems.append(
            f"{label}: run kind {run.get('kind')!r} for shape {run.get('shape')}"
        )
    ms = run.get("milliseconds")
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or ms <= 0:
        problems.append(
            f"{label}: no positive milliseconds for shape {run.get('shape')}"
        )
    if require_trace:
        tc = run.get("trace_check") or {}
        if tc.get("match") is not True or tc.get("totals_match") is not True:
            problems.append(
                f"{label}: trace check missing or not matched for shape {run.get('shape')}"
            )


def same_or_better(value, archived, printed):
    """(verdict, reference, rule) for a latency: at or below the archived per-shape
    timing when one exists, else at or below the printed value plus half an ulp."""
    if archived is not None:
        return (
            ("REPRODUCED" if value <= archived + EPS else "DIFFERS (slower)"),
            archived,
            "archived per-shape timing",
        )
    bound = float(printed) + half_ulp(printed)
    return (
        ("REPRODUCED" if value <= bound + EPS else "DIFFERS (slower)"),
        bound,
        "printed value + half ulp",
    )


def at_least_printed(value, printed):
    bound = float(printed) - half_ulp(printed)
    return ("REPRODUCED" if value >= bound - EPS else "DIFFERS (lower)"), bound


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--manifest", required=True, help="archived tenon_manifest.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sessions = Path(args.sessions_dir)
    paper = {r["case_id"]: r for r in csv.DictReader(open(args.inventory))}
    manifest, attention_shapes = load_manifest(args.manifest)
    problems = []
    report = {
        "kernel_rows": {},
        "score_rows": {},
        "decode": {},
        "simulator": None,
        "diagnostics": {},
    }
    dcc_pairs = []

    def load(name):
        path = sessions / f"attacc_{name}.json"
        if not path.exists():
            problems.append(f"{name}: session missing ({path.name})")
            return None
        data = json.load(open(path))
        if data.get("failures"):
            problems.append(f"{name}: session records failures: {data['failures']}")
        if not isinstance(data.get("runs"), list):
            problems.append(f"{name}: no runs list")
            return None
        if report["simulator"] is None and isinstance(data.get("simulator"), dict):
            report["simulator"] = {
                "name": data["simulator"].get("name"),
                "binary_sha256": data["simulator"].get("binary_sha256"),
                "trace_generator": (data.get("trace_generator") or {}).get(
                    "files_sha256"
                ),
            }
        return data

    # GEMV rows
    for row, (case, nhead, dim, out) in ROWS.items():
        data = load(row)
        if data is None:
            continue
        expected = (nhead, dim, out)
        selected = [r for r in data["runs"] if r.get("selection") == "shared-pipeline"]
        fixed = [r for r in data["runs"] if r.get("selection") == "fixed-form-baseline"]
        others = [
            r
            for r in data["runs"]
            if r.get("selection") not in ("shared-pipeline", "fixed-form-baseline")
        ]
        if len(selected) != 1:
            problems.append(
                f"{row}: {len(selected)} shared-pipeline runs, exactly one required"
            )
        if not fixed:
            problems.append(f"{row}: no fixed-form-baseline run")
        if others:
            problems.append(
                f"{row}: {len(others)} run(s) of another selection: {sorted({r.get('selection') for r in others})}"
            )
        for r in selected:
            check_run(r, row, problems, require_trace=True)
            if shape_key(r) != expected:
                problems.append(
                    f"{row}: selected run shape {r.get('shape')} is not the row's shape {expected}"
                )
        for r in fixed:
            check_run(r, row, problems, require_trace=False)
        if len(selected) != 1 or not fixed:
            continue
        ms = selected[0]["milliseconds"]
        fixed_ms = min(r["milliseconds"] for r in fixed)
        archived = manifest.get(expected)
        archived_ms = (
            float(archived["tenon_time_ms"])
            if archived and archived.get("tenon_time_ms")
            else None
        )
        verdict, reference, rule = same_or_better(
            ms, archived_ms, paper[case]["paper_value"]
        )
        if archived:
            dcc_pairs.append((ms, float(archived["dcc_our_time_ms"])))
        report["kernel_rows"][row] = {
            "case_id": case,
            "selected_ms": ms,
            "paper_ms_printed": paper[case]["paper_value"],
            "archived_tenon_ms": archived_ms,
            "reference": reference,
            "rule": rule,
            "fixed_baseline_ms": fixed_ms,
            "speedup_over_fixed": fixed_ms / ms,
            "tiling": selected[0].get("tiling"),
            "verdict": verdict,
        }
        print(
            f"{row}: {ms:.6f} ms (archived {archived_ms}, printed {paper[case]['paper_value']}); fixed {fixed_ms:.6f}; speedup {fixed_ms/ms:.2f}x -> {verdict}"
        )

    # score rows
    for row, (case, nhead) in SCORE.items():
        data = load(row)
        if data is None:
            continue
        expected = attention_shapes.get(nhead, set())
        if not expected:
            problems.append(
                f"{row}: the archived manifest has no attention shapes for nhead {nhead}"
            )
            continue
        seen, ratios, slower = {}, [], 0
        for r in data["runs"]:
            key = shape_key(r)
            if r.get("selection") != "shared-pipeline":
                problems.append(
                    f"{row}: record of selection {r.get('selection')!r} for shape {r.get('shape')} (only shared-pipeline records are evidence)"
                )
                continue
            check_run(r, row, problems, require_trace=True)
            if key is None or key not in expected:
                problems.append(
                    f"{row}: shape {r.get('shape')} is not one of the row's {len(expected)} archived shapes (nhead {nhead})"
                )
                continue
            if key in seen:
                problems.append(f"{row}: duplicate record for shape {r.get('shape')}")
                continue
            seen[key] = r
        missing = sorted(expected - set(seen))
        if missing:
            problems.append(
                f"{row}: {len(missing)} of {len(expected)} archived shapes missing, first {missing[:3]}"
            )
        for key, r in seen.items():
            m = manifest[key]
            t = r["milliseconds"]
            ratios.append(float(m["fixed_baseline_ms"]) / t)
            if t > float(m["tenon_time_ms"]) + EPS:
                slower += 1
            dcc_pairs.append((t, float(m["dcc_our_time_ms"])))
        figure = first_number(paper[case]["paper_value"], after="Tenon")
        mean_ratio = statistics.mean(ratios) if ratios else float("nan")
        verdict, bound = (
            at_least_printed(mean_ratio, figure)
            if ratios
            else ("DIFFERS (no records)", None)
        )
        if slower:
            verdict = (
                f"DIFFERS ({slower} shape(s) slower than the archived Tenon timing)"
            )
        report["score_rows"][row] = {
            "case_id": case,
            "shapes_expected": len(expected),
            "shapes_present": len(seen),
            "missing": len(missing),
            "mean_ms": (
                statistics.mean(r["milliseconds"] for r in seen.values())
                if seen
                else None
            ),
            "speedup_over_fixed_mean_of_ratios": mean_ratio,
            "figure_printed": figure,
            "bound": bound,
            "shapes_slower_than_archived": slower,
            "verdict": verdict,
        }
        print(
            f"{row}: {len(seen)}/{len(expected)} shapes, speedup over fixed {mean_ratio:.3f}x vs the figure's {figure}x, {slower} slower than archived -> {verdict}"
        )

    # diagnostic tally against DCC over all kernels present
    tally = {"wins": 0, "ties": 0, "losses": 0, "loss_pct": []}
    for t, dcc_ms in dcc_pairs:
        if abs(t - dcc_ms) <= EPS:
            tally["ties"] += 1
        elif t < dcc_ms:
            tally["wins"] += 1
        else:
            tally["losses"] += 1
            tally["loss_pct"].append((t / dcc_ms - 1) * 100)
    tally["median_loss_pct"] = (
        statistics.median(tally["loss_pct"]) if tally["loss_pct"] else None
    )
    tally.pop("loss_pct")
    tally["kernels"] = len(dcc_pairs)
    tally["matches_text"] = (
        len(dcc_pairs) == 500
        and all(tally[k] == TEXT_TALLY[k] for k in ("wins", "ties", "losses"))
        and tally["median_loss_pct"] is not None
        and round(tally["median_loss_pct"], 1) == TEXT_TALLY["median_loss_pct"]
    )
    report["diagnostics"]["tally_vs_dcc"] = tally
    print(
        "diagnostic: tally vs DCC (text 123 / 316 / 61, median loss 7.1 percent):",
        tally,
    )

    # decode
    path = sessions / "attacc_e2e.json"
    if not path.exists():
        problems.append("e2e: composition missing (attacc_e2e.json)")
    else:
        e2e = json.load(open(path))
        if e2e.get("kind") != "analytical-composition":
            problems.append(
                f"e2e: kind {e2e.get('kind')!r}, analytical-composition required"
            )
        configs = e2e.get("configs")
        if not isinstance(configs, list):
            problems.append("e2e: no configs list")
            configs = []
        seen_cfg = {}
        for cfg in configs:
            try:
                key = (int(cfg["lin"]), int(cfg["lout"]), int(cfg["batch"]))
            except (KeyError, TypeError, ValueError):
                problems.append(
                    f"e2e: configuration without lin/lout/batch: {str(cfg)[:80]}"
                )
                continue
            if key not in E2E:
                problems.append(f"e2e: unexpected configuration {key}")
                continue
            if key in seen_cfg:
                problems.append(f"e2e: duplicate configuration {key}")
                continue
            seen_cfg[key] = cfg
        for key in E2E:
            if key not in seen_cfg:
                problems.append(f"e2e: configuration {key} missing")
        for key, cfg in seen_cfg.items():
            case = E2E[key]
            tenon = (cfg.get("variants") or {}).get("tenon") or {}
            base, full = tenon.get("base_speedup"), tenon.get("full_speedup")
            if not isinstance(base, (int, float)) or not isinstance(full, (int, float)):
                problems.append(f"e2e: {case} has no tenon base/full speedup")
                continue
            printed = paper[case]["paper_value"]
            pb, pf = first_number(printed, after="base"), first_number(
                printed, after="full"
            )
            vb, bb = at_least_printed(base, pb)
            vf, bf = at_least_printed(full, pf)
            verdict = (
                "REPRODUCED"
                if vb == vf == "REPRODUCED"
                else f"DIFFERS (base {vb}, full {vf})"
            )
            report["decode"][case] = {
                "case_id": case,
                "base_speedup": base,
                "full_speedup": full,
                "paper_printed": printed,
                "base_bound": bb,
                "full_bound": bf,
                "verdict": verdict,
                "dcc_full": ((cfg.get("variants") or {}).get("dcc") or {}).get(
                    "full_speedup"
                ),
                "fixed_full": ((cfg.get("variants") or {}).get("fixed") or {}).get(
                    "full_speedup"
                ),
            }
            print(
                f"{case}: base {base:.2f}x full {full:.2f}x vs paper {printed} -> {verdict}"
            )

    expected_cases = (
        [v[0] for v in ROWS.values()]
        + [v[0] for v in SCORE.values()]
        + list(E2E.values())
    )
    verdicts = {}
    for section in ("kernel_rows", "score_rows", "decode"):
        for entry in report[section].values():
            verdicts[entry["case_id"]] = entry["verdict"]
    missing_verdicts = [c for c in expected_cases if c not in verdicts]
    if missing_verdicts:
        problems.append(
            f"{len(missing_verdicts)} of 12 result verdicts missing: {missing_verdicts}"
        )
    report["problems"] = problems
    report["verdicts"] = verdicts
    report["all_reproduced"] = (
        not problems
        and len(verdicts) == 12
        and all(v == "REPRODUCED" for v in verdicts.values())
    )
    json.dump(report, open(args.out, "w"), indent=1)
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    print("ALL REPRODUCED" if report["all_reproduced"] else "NOT ALL REPRODUCED")
    return 0 if report["all_reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
