#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool Gemini-I (APU v1) paper-row sessions under the fixed acceptance rule.

Round 142 (ISSUES X-46): the round 141 pooler trusted each session's cached
``statistics.samples`` and ``measured_crun`` and never looked at the launch
records, so a session with no launches and cached ten-sample claims pooled
as REPRODUCED.  This pooler derives every sample from the launch records
and validates them; the cached statistics must agree with the derivation
or the row is incomplete.

Per row (case) of a session (``benchmarks/apu_v1/paper_rows.py`` output):
every required lane has exactly ``--warmups`` warm-up launches and
``--measured`` measured launches whose rounds are exactly 1..N once each
(the lanes interleaved round by round); every launch of those lanes ran
without error, matched the oracle, and carries a positive ``crun`` equal to
its profiler ``counters.crun`` with ``hits`` 1 and positive counters; the
every launch carries a binary identity, the archived lanes' launches all
the lane's declared archived binary (``binary_matches_archive``) and the
Tenon lane's launches its declared launched source and route (the Tenon
binary is rebuilt per launch and is not bit-reproducible, so its identity
is the source and route; every binary hash is kept); every lane records
its fixture (``inputs_sha256``: each lane reads its own archived input
images) and the lanes share the oracle (``oracle_sha256``) and the board
lock (recorded per launch by the Tenon runtime, per record for the
archived lanes); the cached ``measured_crun`` and ``statistics`` equal the
derived samples; no problem is recorded for the row.  A row that fails any of this
is reported with its reasons and excluded; the other rows of the same
session still pool (Gemini-I's per-row completeness rule).

Only a default-selection row counts as reproduction evidence: its label
is ``default_selection``, the session records no cost perturbation, the
Tenon lane declares no substituted source, and its launched source is the
compiler-selected source (``launched_source_sha256`` equal to
``device_source_sha256``, both valid); its route is the lane's native
route (every launch's ``route`` equal to it) or its selected vector plan;
its cost fingerprint is recorded.  A row that is a labelled control (a
historical or substituted program, a forced plan, a perturbed cost) is
validated the same way but reported under ``controls`` and never pooled.
The launch sequence must be the driver's: every lane's warm-up in the
protocol's lane order, then rounds 1..N with the lanes in that order on
odd rounds and reversed on even rounds.

Across sessions: the installed package path and host are checked against
the first session, and each row is checked against its first complete
occurrence in any session (so a row incomplete in the first session is
still compared between the later sessions): protocol, paper values,
fixtures, oracle, Tenon source, route and cost fingerprint, archived
binaries and board lock must be identical; two sessions presenting the
same launch directories or the same measured evidence are duplicates and
both are refused.

Rule (``docs/release/PAPER_REPRODUCTION.md``, round 31): the row's
statistic is the median of the pooled Tenon samples of its complete rows at
the revision; REPRODUCED when at or below the paper's value (every launch
exact by validation), OPEN (nominal) when the paper's value lies inside the
pooled interquartile range, OPEN (regression) when it lies below it.
``--quota N`` records the pre-registered number of complete sessions per
row: the pooled statistics are reported whatever the count, and the exit
status is 3 while any row has fewer complete sessions than the quota.

Exit status: 0 pooled and every row at quota; 2 a session file invalid,
duplicate or incompatible (nothing pooled unless ``--exclude-invalid``,
which pools the rest and lists the rejected files); 3 pooled but a row
below quota.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

_SHA = re.compile(r"^[0-9a-f]{64}$")


class SessionInvalid(Exception):
    """A session file that cannot be pooled at all."""


def _require(condition, message):
    if not condition:
        raise SessionInvalid(message)


def quartiles(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {"samples": 0}
    lower = xs[: n // 2]
    upper = xs[(n + 1) // 2 :]
    return {
        "samples": n,
        "median": statistics.median(xs),
        "q1": statistics.median(lower) if lower else xs[0],
        "q3": statistics.median(upper) if upper else xs[-1],
        "min": xs[0],
        "max": xs[-1],
    }


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha(value):
    return isinstance(value, str) and bool(_SHA.match(value))


DEFAULT_LABEL = "default_selection"


def row_kind(record, session_perturbation=None):
    """``"default"`` for a default-selection row of an unperturbed session,
    else ``"control"`` with the reason (a labelled control, a substituted
    program or a perturbed cost program)."""
    label = record.get("label")
    tenon = (record.get("lanes") or {}).get("tenon") or {}
    reasons = []
    if label != DEFAULT_LABEL:
        reasons.append(f"label {label!r}")
    if tenon.get("substituted_source") not in (None, {}):
        reasons.append("a substituted source")
    if session_perturbation not in (None, "None", {}):
        reasons.append(f"a perturbed cost program {session_perturbation!r}")
    return ("control", "; ".join(reasons)) if reasons else ("default", "")


def expected_launch_sequence(lanes, measured, warmups):
    """The driver's launch order (``paper_rows.py``): every lane's warm-ups
    in the protocol's lane order, then rounds 1..N with the lanes in that
    order on odd rounds and reversed on even rounds."""
    sequence = []
    for _ in range(warmups):
        for lane in lanes:
            sequence.append((lane, "warmup", 0))
    for rnd in range(1, measured + 1):
        order = list(lanes) if rnd % 2 == 1 else list(reversed(lanes))
        for lane in order:
            sequence.append((lane, "measured", rnd))
    return sequence


def validate_row(
    record, *, measured, warmups, required_lanes, session_perturbation=None
):
    """The derived evidence of one row, or a list of reasons it is incomplete.

    Returns ``(evidence, problems)``: ``evidence`` is None when ``problems``
    is non-empty.  ``evidence["kind"]`` is ``"default"`` or ``"control"``.
    """
    problems = []
    case = record.get("case")
    protocol = record.get("protocol") or {}
    lanes = record.get("lanes") or {}
    launches = record.get("launches")
    if not isinstance(launches, list) or not launches:
        return None, [f"{case}: no launch records"]
    if record.get("problems"):
        problems.append(f"{case}: problems recorded: {record['problems']!r}")
    if protocol.get("measured_per_lane") != measured:
        problems.append(
            f"{case}: protocol measured_per_lane {protocol.get('measured_per_lane')!r}, {measured} required"
        )
    if protocol.get("warmups_per_lane") != warmups:
        problems.append(
            f"{case}: protocol warmups_per_lane {protocol.get('warmups_per_lane')!r}, {warmups} required"
        )
    for lane in required_lanes:
        if lane not in lanes:
            problems.append(f"{case}: required lane {lane!r} absent")
        if lane not in (protocol.get("lanes") or ()):
            problems.append(f"{case}: lane {lane!r} not in the protocol's lanes")
    if problems:
        return None, problems
    protocol_lanes = list(protocol.get("lanes") or ())
    actual = [(l.get("lane"), l.get("kind"), l.get("round")) for l in launches]
    expected = expected_launch_sequence(protocol_lanes, measured, warmups)
    if actual != expected:
        first = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        problems.append(
            f"{case}: the launch sequence is not the driver's interleaved order "
            f"(first difference at launch {first}: recorded "
            f"{actual[first] if first < len(actual) else None}, expected "
            f"{expected[first] if first < len(expected) else None}; "
            f"{len(actual)} recorded, {len(expected)} expected)"
        )
        return None, problems
    paper = record.get("paper_tenon_cycles")
    if not isinstance(paper, (int, float)) or isinstance(paper, bool) or paper <= 0:
        return None, [f"{case}: paper_tenon_cycles {paper!r}"]
    oracle = lanes[required_lanes[0]].get("oracle_sha256")
    if not _sha(oracle):
        problems.append(f"{case}: no oracle identity (oracle_sha256)")
    inputs = {}
    for lane in required_lanes:
        fixture = lanes[lane].get("inputs_sha256")
        if (
            not isinstance(fixture, dict)
            or not fixture
            or not all(_sha(v) for v in fixture.values())
        ):
            problems.append(f"{case}/{lane}: no fixture identity (inputs_sha256)")
        inputs[lane] = fixture
        if lanes[lane].get("oracle_sha256") != oracle:
            problems.append(f"{case}: lane {lane!r} records a different oracle")
    samples = {}
    by_round = {}
    binaries = {}
    launch_dirs = []
    locks = set()
    for lane in required_lanes:
        mine = [l for l in launches if l.get("lane") == lane]
        warm = [l for l in mine if l.get("kind") == "warmup"]
        meas = [l for l in mine if l.get("kind") == "measured"]
        if len(warm) != warmups:
            problems.append(
                f"{case}/{lane}: {len(warm)} warm-up launches, {warmups} required"
            )
        rounds = sorted(l.get("round") for l in meas)
        if rounds != list(range(1, measured + 1)):
            problems.append(
                f"{case}/{lane}: measured rounds {rounds}, rounds 1..{measured} once each required"
            )
        lane_binaries = set()
        for l in warm + meas:
            where = f"{case}/{lane} {l.get('kind')} round {l.get('round')}"
            if l.get("error"):
                problems.append(f"{where}: failed launch ({str(l['error'])[:80]})")
                continue
            if l.get("outputs_match_oracle") is not True:
                problems.append(
                    f"{where}: outputs_match_oracle is {l.get('outputs_match_oracle')!r}"
                )
            crun = l.get("crun")
            if not _positive_int(crun):
                problems.append(f"{where}: crun {crun!r} is not a positive integer")
            counters = l.get("counters")
            if not isinstance(counters, dict) or not counters:
                problems.append(f"{where}: no profiler counters")
            else:
                if counters.get("hits") != 1:
                    problems.append(
                        f"{where}: profiler hits {counters.get('hits')!r}, one measurement required"
                    )
                if counters.get("crun") != crun:
                    problems.append(
                        f"{where}: profiler counters.crun {counters.get('crun')!r} differs from the recorded crun {crun!r}"
                    )
                for key in ("seu", "iall"):
                    if not _positive_int(counters.get(key)):
                        problems.append(
                            f"{where}: profiler {key} {counters.get(key)!r} is not a positive integer"
                        )
            if not _sha(l.get("binary_sha256")):
                problems.append(f"{where}: no binary identity")
            else:
                lane_binaries.add(l["binary_sha256"])
            lock = l.get("board_lock") or (
                record.get("board_lock") if lane != "tenon" else None
            )
            if lock:
                locks.add(lock)
            else:
                problems.append(f"{where}: no board lock recorded")
            if l.get("launch_dir"):
                launch_dirs.append(l["launch_dir"])
            if lane == "tenon":
                if not _sha(l.get("launched_source_sha256")) or l.get(
                    "launched_source_sha256"
                ) != lanes["tenon"].get("launched_source_sha256"):
                    problems.append(
                        f"{where}: launched source "
                        f"{str(l.get('launched_source_sha256'))[:12]!r} is not the "
                        "lane's declared launched source"
                    )
                native = lanes["tenon"].get("native_vector_route")
                if native is not None and l.get("route") != native:
                    problems.append(
                        f"{where}: route {l.get('route')!r} differs from the lane's "
                        f"native route {native!r}"
                    )
            if l.get("kind") == "measured" and _positive_int(crun):
                by_round.setdefault(l.get("round"), {})[lane] = crun
        if lane != "tenon" and len(lane_binaries) > 1:
            problems.append(
                f"{case}/{lane}: several binaries launched {sorted(lane_binaries)}"
            )
        if lane != "tenon":
            declared = lanes[lane].get("binary_sha256")
            if lanes[lane].get("binary_matches_archive") is not True:
                problems.append(
                    f"{case}/{lane}: binary_matches_archive is {lanes[lane].get('binary_matches_archive')!r}"
                )
            if lane_binaries and declared not in lane_binaries:
                problems.append(
                    f"{case}/{lane}: launched binary differs from the declared archived binary"
                )
        binaries[lane] = sorted(lane_binaries)
        derived = [
            l["crun"]
            for l in sorted(meas, key=lambda l: l.get("round") or 0)
            if _positive_int(l.get("crun")) and not l.get("error")
        ]
        samples[lane] = derived
        cached = lanes[lane].get("measured_crun")
        stats = lanes[lane].get("statistics") or {}
        if list(cached or []) != derived:
            problems.append(
                f"{case}/{lane}: cached measured_crun differs from the launch records"
            )
        if derived and (
            stats.get("samples") != len(derived)
            or stats.get("median") != statistics.median(derived)
        ):
            problems.append(
                f"{case}/{lane}: cached statistics differ from the launch records"
            )
    if len(locks) > 1:
        problems.append(f"{case}: launches under different board locks {sorted(locks)}")
    # The Tenon lane's provenance: the compiler-selected source, the launched
    # source bound to it, exactly one route form, the cost fingerprint.
    tenon = lanes["tenon"]
    selected_source = tenon.get("device_source_sha256")
    launched_source = tenon.get("launched_source_sha256")
    if not _sha(selected_source):
        problems.append(
            f"{case}/tenon: no valid compiler-selected source identity (device_source_sha256)"
        )
    if not _sha(launched_source):
        problems.append(
            f"{case}/tenon: no valid launched source identity (launched_source_sha256)"
        )
    kind, kind_reason = row_kind(record, session_perturbation)
    if (
        kind == "default"
        and _sha(selected_source)
        and launched_source != selected_source
    ):
        problems.append(
            f"{case}/tenon: the launched source is not the compiler-selected source"
        )
    native = tenon.get("native_vector_route")
    plan = tenon.get("selected_plan")
    if bool(native) == bool(plan):
        problems.append(
            f"{case}/tenon: exactly one route form is required "
            f"(native_vector_route {native!r}, selected_plan {plan!r})"
        )
    elif not isinstance(native or plan, str):
        problems.append(f"{case}/tenon: the route identity is not a name")
    if not isinstance(tenon.get("cost_fingerprint"), str) or not tenon.get(
        "cost_fingerprint"
    ):
        problems.append(f"{case}/tenon: no cost fingerprint")
    if problems:
        return None, problems
    paired = []
    for r in range(1, measured + 1):
        lanes_in_round = by_round.get(r, {})
        if set(lanes_in_round) != set(required_lanes):
            return None, [f"{case}: round {r} lacks a launch of every required lane"]
        if "expert" in lanes_in_round:
            paired.append(lanes_in_round["expert"] / lanes_in_round["tenon"])
    evidence = {
        "case": case,
        "kind": kind,
        "kind_reason": kind_reason,
        "label": record.get("label"),
        "paper_tenon": paper,
        "paper_expert": record.get("paper_expert_cycles"),
        "protocol": protocol,
        "inputs": inputs,
        "oracle": oracle,
        "board_lock": sorted(locks)[0] if locks else None,
        "tenon_source": launched_source,
        "selected_source": selected_source,
        "tenon_route": native or plan,
        "cost_fingerprint": lanes["tenon"].get("cost_fingerprint"),
        "binaries": binaries,
        "samples": samples,
        "paired_expert_over_tenon": paired,
        "launch_dirs": launch_dirs,
        "evidence_key": tuple(
            sorted(
                (lane, r, c)
                for r, lanes_in_round in by_round.items()
                for lane, c in lanes_in_round.items()
            )
        ),
    }
    return evidence, []


def validate_session(d, *, measured, warmups, required_lanes):
    """The session's complete rows (and the incomplete ones with reasons)."""
    for key in ("host", "allo", "records"):
        _require(key in d, f"missing field {key!r}")
    _require(isinstance(d["records"], list) and d["records"], "no records")
    rows, controls, incomplete = {}, {}, {}
    for record in d["records"]:
        case = record.get("case")
        _require(isinstance(case, str) and case, "a record without a case name")
        _require(
            case not in rows and case not in incomplete and case not in controls,
            f"two records for {case!r}",
        )
        evidence, problems = validate_row(
            record,
            measured=measured,
            warmups=warmups,
            required_lanes=required_lanes,
            session_perturbation=d.get("perturbation"),
        )
        if evidence is None:
            incomplete[case] = problems
        elif evidence["kind"] == "control":
            controls[case] = evidence
        else:
            rows[case] = evidence
    return {
        "host": d["host"],
        "allo": d["allo"],
        "date": d.get("date"),
        "status": d.get("status"),
        "perturbation": d.get("perturbation"),
        "rows": rows,
        "controls": controls,
        "incomplete": incomplete,
    }


ROW_IDENTITY = (
    "paper_tenon",
    "paper_expert",
    "protocol",
    "inputs",
    "oracle",
    "tenon_source",
    "selected_source",
    "tenon_route",
    "cost_fingerprint",
    "board_lock",
)


EQUIVALENCE_SCHEMA = "tenon-g1-installation-equivalence-v1"


def _package_root(allo_init):
    """The installed ``allo`` package directory of a recorded ``allo.__file__``."""
    import os

    root = os.path.dirname(allo_init)
    _require(
        os.path.basename(allo_init) == "__init__.py" and os.path.basename(root) == "allo",
        f"not an installed allo package path: {allo_init!r}",
    )
    return root


def _installation_hashes(root):
    """sha256 of every file of an installed package (no bytecode caches)."""
    import hashlib, os

    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _file_sha256(path):
    import hashlib

    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_equivalence(out_path, installations, allowed, substitutions, decision):
    """Write an installation-equivalence manifest from live artifacts.

    ``installations``: ``[(allo_init_path, wheel_path, revision), ...]``;
    ``allowed``: ``[(relative file, reason), ...]`` the reviewed compiler
    delta allows to differ; ``substitutions``: ``[(old, new), ...]`` byte
    strings whose substitution must make an allowed file identical (embedded
    build paths).  Every file of every installation is hashed now; the pooler
    re-hashes at use and refuses any drift."""
    manifest = {
        "schema": EQUIVALENCE_SCHEMA,
        "decision": decision,
        "installations": [],
        "allowed_differences": [{"file": f, "reason": r} for f, r in allowed],
        "substitutions": [[a, b] for a, b in substitutions],
    }
    for allo_init, wheel, revision in installations:
        root = _package_root(allo_init)
        import os

        manifest["installations"].append(
            {
                "allo": allo_init,
                "allo_realpath": os.path.realpath(allo_init),
                "revision": revision,
                "wheel_path": wheel,
                "wheel_sha256": _file_sha256(wheel),
                "files": _installation_hashes(root),
            }
        )
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def load_equivalence(path):
    with open(path) as fh:
        manifest = json.load(fh)
    _require(
        manifest.get("schema") == EQUIVALENCE_SCHEMA,
        f"equivalence manifest {path!r}: unknown schema {manifest.get('schema')!r}",
    )
    _require(
        bool(manifest.get("decision")),
        f"equivalence manifest {path!r}: no reviewed decision recorded",
    )
    return manifest


def _verify_installation(manifest, allo_init):
    """The manifest's record of this installation, re-verified against the
    live files and the wheel; a path the manifest does not list is refused."""
    import os

    # Installations are identified by their resolved path: a campaign
    # directory's pkg is a symlink to the package directory, so sessions
    # record the symlink's path; the files hashed are the resolved files.
    entries = [
        e
        for e in manifest.get("installations", [])
        if os.path.realpath(e.get("allo", "")) == os.path.realpath(allo_init)
    ]
    _require(
        len(entries) == 1,
        f"installed package {allo_init!r} (resolved {os.path.realpath(allo_init)!r}) "
        "is not listed in the equivalence manifest",
    )
    entry = entries[0]
    _require(
        os.path.isfile(entry.get("wheel_path", "")),
        f"the wheel of {allo_init!r} is missing: {entry.get('wheel_path')!r}",
    )
    _require(
        _file_sha256(entry["wheel_path"]) == entry.get("wheel_sha256"),
        f"the wheel of {allo_init!r} does not match the manifest's sha256",
    )
    live = _installation_hashes(_package_root(allo_init))
    _require(
        live == entry.get("files"),
        f"installed package {allo_init!r} differs from the manifest's recorded "
        "files (the installation drifted or the manifest is stale)",
    )
    return entry


def _equal_after_substitution(path_a, path_b, substitutions):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        a, b = fa.read(), fb.read()
    for old, new in substitutions:
        b = b.replace(new.encode(), old.encode())
        a = a.replace(new.encode(), old.encode())
    return a == b


def installations_equivalent(manifest, allo_a, allo_b):
    """Both installations re-verified against the manifest; every file that
    differs between them must be an allowed difference of the reviewed
    delta, and an allowed file covered by a substitution must be identical
    after it.  Raises ``SessionInvalid`` otherwise."""
    import os

    if allo_a == allo_b:
        return True
    ea, eb = _verify_installation(manifest, allo_a), _verify_installation(manifest, allo_b)
    allowed = {d["file"]: d.get("reason", "") for d in manifest.get("allowed_differences", [])}
    files_a, files_b = ea["files"], eb["files"]
    _require(
        set(files_a) == set(files_b),
        "the two installations do not have the same files: "
        f"only in one: {sorted(set(files_a) ^ set(files_b))[:5]}",
    )
    differing = sorted(f for f in files_a if files_a[f] != files_b[f])
    unexpected = [f for f in differing if f not in allowed]
    _require(
        not unexpected,
        f"files differ between {allo_a!r} and {allo_b!r} outside the reviewed "
        f"delta: {unexpected[:5]}",
    )
    substitutions = [tuple(s) for s in manifest.get("substitutions", [])]
    for f in differing:
        pa = os.path.join(_package_root(allo_a), f)
        pb = os.path.join(_package_root(allo_b), f)
        if substitutions and not f.endswith(".py"):
            _require(
                _equal_after_substitution(pa, pb, substitutions),
                f"allowed file {f!r} differs beyond the recorded substitutions",
            )
    return True


_EQUIVALENCE = {"manifest": None, "verified": set()}


def set_equivalence_manifest(manifest):
    _EQUIVALENCE["manifest"] = manifest
    _EQUIVALENCE["verified"] = set()


def check_compatible(reference, other, label):
    """The session-level identities against the first session.  A different
    installed package is accepted only when an equivalence manifest lists
    both installations and the live artifacts verify as the reviewed delta
    allows (``--equivalence``); otherwise it is refused as before."""
    import os

    same_installation = reference["allo"] == other["allo"] or (
        os.path.realpath(reference["allo"]) == os.path.realpath(other["allo"])
    )
    if not same_installation:
        manifest = _EQUIVALENCE["manifest"]
        _require(
            manifest is not None,
            f"{label}: installed package {other['allo']!r} differs from the first "
            f"session's {reference['allo']!r} (no equivalence manifest given)",
        )
        key = tuple(sorted((reference["allo"], other["allo"])))
        if key not in _EQUIVALENCE["verified"]:
            try:
                installations_equivalent(manifest, reference["allo"], other["allo"])
            except SessionInvalid as exc:
                raise SessionInvalid(f"{label}: {exc}") from exc
            _EQUIVALENCE["verified"].add(key)
    _require(
        reference["host"] == other["host"],
        f"{label}: host differs from the first session",
    )


def check_rows_compatible(validated, labels):
    """Every complete occurrence of a row (default or control) against the
    row's first complete occurrence in any session, whatever the input
    order; the reference holder is named in the message."""
    reference = {}
    for v, label in zip(validated, labels):
        for group in ("rows", "controls"):
            for case, e in v[group].items():
                # default rows are compared with default rows; a labelled
                # control (another program by design) with controls of the
                # same label only
                key = (case, group, e["label"] if group == "controls" else None)
                ref = reference.get(key)
                if ref is None:
                    reference[key] = (e, label)
                    continue
                a, held = ref
                for key in ROW_IDENTITY:
                    _require(
                        a[key] == e[key],
                        f"{label}: {case} {key} differs from its first complete "
                        f"occurrence in {held!r}",
                    )
                for lane, hashes in a["binaries"].items():
                    if lane != "tenon":
                        _require(
                            hashes == e["binaries"].get(lane),
                            f"{label}: {case} {lane} archived binary differs from "
                            f"its first complete occurrence in {held!r}",
                        )


def check_duplicates(validated, labels):
    seen_dirs, seen_evidence = {}, {}
    for v, label in zip(validated, labels):
        for case, e in list(v["rows"].items()) + list(v["controls"].items()):
            for directory in e["launch_dirs"]:
                _require(
                    directory not in seen_dirs,
                    f"{label}: {case} presents launch directory {directory!r} already presented by {seen_dirs.get(directory)!r} (duplicate evidence)",
                )
                seen_dirs[directory] = label
            _require(
                e["evidence_key"] not in seen_evidence,
                f"{label}: {case} presents the same measured evidence as {seen_evidence.get(e['evidence_key'])!r} (duplicate evidence)",
            )
            seen_evidence[e["evidence_key"]] = label


def pool(validated, labels, *, quota, row_quota=None):
    rows = {}
    cases = []
    for v in validated:
        for case in list(v["rows"]) + list(v["incomplete"]) + list(v["controls"]):
            if case not in cases:
                cases.append(case)
    for case in cases:
        case_quota = (row_quota or {}).get(case, quota)
        complete = [
            (v, label) for v, label in zip(validated, labels) if case in v["rows"]
        ]
        incomplete = {
            label: v["incomplete"][case]
            for v, label in zip(validated, labels)
            if case in v["incomplete"]
        }
        controls = {
            label: {
                "label": v["controls"][case]["label"],
                "reason": v["controls"][case]["kind_reason"],
                "launched_source_sha256": v["controls"][case]["tenon_source"],
                "selected_source_sha256": v["controls"][case]["selected_source"],
                "tenon_median": statistics.median(
                    v["controls"][case]["samples"]["tenon"]
                ),
            }
            for v, label in zip(validated, labels)
            if case in v["controls"]
        }
        if not complete:
            rows[case] = {
                "complete_sessions": 0,
                "quota": case_quota,
                "remaining": case_quota,
                "incomplete_sessions": incomplete,
                "controls": controls,
                "status": "no complete session",
            }
            continue
        ref = complete[0][0]["rows"][case]
        tenon, expert, historical, paired, medians = [], [], [], [], []
        for v, label in complete:
            e = v["rows"][case]
            tenon += e["samples"]["tenon"]
            expert += e["samples"].get("expert", [])
            historical += e["samples"].get("historical_binary", [])
            paired += e["paired_expert_over_tenon"]
            medians.append(statistics.median(e["samples"]["tenon"]))
        t = quartiles(tenon)
        paper = ref["paper_tenon"]
        if t["median"] <= paper:
            status = "REPRODUCED"
        elif t["q1"] <= paper <= t["q3"]:
            status = "OPEN (nominal)"
        else:
            status = "OPEN (regression)"
        rows[case] = {
            "complete_sessions": len(complete),
            "quota": case_quota,
            "remaining": max(0, case_quota - len(complete)),
            "sessions": [label for _v, label in complete],
            "incomplete_sessions": incomplete,
            "controls": controls,
            "paper_tenon": paper,
            "paper_expert": ref["paper_expert"],
            "pooled_tenon": t,
            "ratio_to_paper": t["median"] / paper,
            "session_medians": medians,
            "pooled_expert": quartiles(expert),
            "pooled_historical_binary": quartiles(historical),
            "paired_expert_over_tenon": quartiles(paired),
            "paper_expert_over_tenon": (
                (ref["paper_expert"] / paper) if ref["paper_expert"] else None
            ),
            "tenon_source_sha256": ref["tenon_source"],
            "selected_source_sha256": ref["selected_source"],
            "tenon_route": ref["tenon_route"],
            "cost_fingerprint": ref["cost_fingerprint"],
            "binaries": ref["binaries"],
            "inputs_sha256": ref["inputs"],
            "oracle_sha256": ref["oracle"],
            "outputs_exact": True,
            "status": status,
        }
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="the pooled result (required unless --write-equivalence)")
    ap.add_argument("--measured", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--required-lanes", default="tenon,expert,historical_binary")
    ap.add_argument(
        "--quota", type=int, default=3, help="pre-registered complete sessions per row"
    )
    ap.add_argument(
        "--row-quota",
        default="",
        help="per-row quotas overriding --quota, e.g. kmeans=3,word_count=3,linear_regression=3",
    )
    ap.add_argument(
        "--plan-out",
        default=None,
        help="also write the remaining per-row plan (complete rows still owed, the cases of the next session)",
    )
    ap.add_argument(
        "--exclude-invalid",
        action="store_true",
        help="pool the valid sessions and list the rejected files instead of failing",
    )
    ap.add_argument(
        "--equivalence",
        default=None,
        help="installation-equivalence manifest (written by --write-equivalence) that lets "
        "sessions from two verified-equivalent installed packages pool together",
    )
    ap.add_argument(
        "--preflight-install",
        default=None,
        help="an installed allo/__init__.py path: check that it may pool with every given "
        "session (identical, or verified equivalent through --equivalence) and exit; no pooling",
    )
    ap.add_argument(
        "--write-equivalence",
        default=None,
        help="write an equivalence manifest to this path from --installation, --allow, "
        "--substitution and --decision, hashing the live artifacts; no pooling",
    )
    ap.add_argument(
        "--installation",
        action="append",
        default=[],
        help="ALLO_INIT_PATH:WHEEL_PATH:REVISION (repeatable; with --write-equivalence)",
    )
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        help="RELATIVE_FILE=REASON: a file the reviewed compiler delta allows to differ",
    )
    ap.add_argument(
        "--substitution",
        action="append",
        default=[],
        help="OLD=NEW: byte strings whose substitution makes an allowed non-Python file identical",
    )
    ap.add_argument("--decision", default="", help="the reviewed decision the manifest records")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args(argv)
    if args.write_equivalence:
        if not args.installation or not args.decision:
            print("--write-equivalence needs --installation entries and --decision", file=sys.stderr)
            return 2
        installations = []
        for item in args.installation:
            parts = item.split(":")
            if len(parts) != 3:
                print(f"--installation expects ALLO_INIT_PATH:WHEEL_PATH:REVISION, got {item!r}", file=sys.stderr)
                return 2
            installations.append(tuple(parts))
        allowed = [tuple(a.split("=", 1)) for a in args.allow]
        substitutions = [tuple(s.split("=", 1)) for s in args.substitution]
        try:
            manifest = write_equivalence(args.write_equivalence, installations, allowed, substitutions, args.decision)
        except (SessionInvalid, OSError) as exc:
            print(f"equivalence manifest not written: {exc}", file=sys.stderr)
            return 2
        for entry in manifest["installations"]:
            print(f"installation {entry['allo']}: {len(entry['files'])} files, wheel {entry['wheel_sha256'][:16]}, revision {entry['revision']}")
        print(f"wrote {args.write_equivalence}")
        return 0
    if not args.files:
        print("session files are required", file=sys.stderr)
        return 2
    if not args.out:
        print("--out is required", file=sys.stderr)
        return 2
    if args.equivalence:
        try:
            set_equivalence_manifest(load_equivalence(args.equivalence))
        except (SessionInvalid, OSError, ValueError) as exc:
            print(f"equivalence manifest refused: {exc}", file=sys.stderr)
            return 2
    if args.preflight_install:
        problems = []
        for f in args.files:
            try:
                with open(f) as fh:
                    d = json.load(fh)
                check_compatible(
                    {"allo": d["allo"], "host": d["host"]},
                    {"allo": args.preflight_install, "host": d["host"]},
                    f"{f} against {args.preflight_install}",
                )
                print(f"compatible: {f} ({d['allo']})")
            except (SessionInvalid, OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(str(exc))
                print(f"INCOMPATIBLE: {exc}", file=sys.stderr)
        if problems:
            print(f"preflight: {len(problems)} session file(s) cannot pool with {args.preflight_install}; nothing launched", file=sys.stderr)
            return 2
        print(f"preflight: {args.preflight_install} may pool with all {len(args.files)} session file(s)")
        return 0
    lanes = tuple(x for x in args.required_lanes.split(",") if x)
    if not lanes or lanes[0] != "tenon":
        print("the required lanes must start with 'tenon'", file=sys.stderr)
        return 2
    row_quota = {}
    for item in (x for x in args.row_quota.split(",") if x):
        case, _eq, value = item.partition("=")
        row_quota[case.strip()] = int(value)
    validated, labels, rejected = [], [], []
    for f in args.files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            v = validate_session(
                d, measured=args.measured, warmups=args.warmups, required_lanes=lanes
            )
            if validated:
                check_compatible(validated[0], v, f)
            check_rows_compatible(validated + [v], labels + [f])
            check_duplicates(validated + [v], labels + [f])
            validated.append(v)
            labels.append(f)
        except (SessionInvalid, OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({"file": f, "reason": str(exc)})
            print(f"REJECTED {f}: {exc}", file=sys.stderr)
    if rejected and not args.exclude_invalid:
        print(
            f"{len(rejected)} session file(s) rejected; nothing pooled", file=sys.stderr
        )
        return 2
    if not validated:
        print("no valid session", file=sys.stderr)
        return 2
    rows = pool(validated, labels, quota=args.quota, row_quota=row_quota)
    for case in row_quota:
        rows.setdefault(
            case,
            {
                "complete_sessions": 0,
                "quota": row_quota[case],
                "remaining": row_quota[case],
                "incomplete_sessions": {},
                "controls": {},
                "status": "no complete session",
            },
        )
    below = [case for case, r in rows.items() if r["complete_sessions"] < r["quota"]]
    plan = {
        "remaining_complete_rows": {
            case: r["remaining"] for case, r in rows.items() if r["remaining"]
        },
        "next_session_cases": [case for case, r in rows.items() if r["remaining"]],
        "pooled_files": list(labels),
    }
    if args.plan_out:
        with open(args.plan_out, "w") as fh:
            json.dump(plan, fh, indent=1)
    for case, r in rows.items():
        if r["complete_sessions"]:
            t = r["pooled_tenon"]
            print(
                f"{case}: {r['complete_sessions']}/{args.quota} complete sessions, pooled median {t['median']} / paper {r['paper_tenon']} ratio {r['ratio_to_paper']:.6f} IQR [{t['q1']}, {t['q3']}] -> {r['status']}"
            )
        else:
            print(f"{case}: 0/{args.quota} complete sessions -> {r['status']}")
        for label, problems in r["incomplete_sessions"].items():
            print(
                f"   incomplete in {label}: {problems[0][:200]}"
                + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
            )
        for label, control in r.get("controls", {}).items():
            print(
                f"   control (not pooled) in {label}: {control['label']} "
                f"({control['reason']}), tenon median {control['tenon_median']}"
            )
    with open(args.out, "w") as fh:
        json.dump(
            {
                "rule": "pooled median of the complete rows at this revision; REPRODUCED if <= paper with every launch exact; OPEN (nominal) if the paper's value lies inside the pooled IQR; OPEN (regression) if below it",
                "protocol": {
                    "measured": args.measured,
                    "warmups": args.warmups,
                    "required_lanes": list(lanes),
                    "quota": args.quota,
                },
                "sessions": [
                    {
                        "file": label,
                        "host": v["host"],
                        "date": v["date"],
                        "allo": v["allo"],
                        "status": v["status"],
                        "complete_rows": sorted(v["rows"]),
                        "incomplete_rows": {c: p for c, p in v["incomplete"].items()},
                    }
                    for v, label in zip(validated, labels)
                ],
                "rejected_sessions": rejected,
                "rows": rows,
                "plan": plan,
                "campaign_complete": not below,
            },
            fh,
            indent=1,
            default=str,
        )
    if below:
        print(
            f"CAMPAIGN INCOMPLETE: rows below their quota: {plan['remaining_complete_rows']}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
