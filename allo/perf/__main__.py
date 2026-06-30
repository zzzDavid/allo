# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CLI for agent-driven calibration updates.

Examples::

    python -m allo.perf validate calibration.json
    python -m allo.perf fit --base base.json --probes probes.json \
        --measurements measurements.jsonl --name board/fw --output fitted.json
    python -m allo.perf diff base.json fitted.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from .calibration import (
    CalibrationProfile,
    MeasurementRecord,
    ProbeSpec,
    fit_profile,
)


def _load_records(path, cls):
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for row in rows:
        if cls is MeasurementRecord:
            row = {**row, "cycles_samples": tuple(row["cycles_samples"])}
        out.append(cls(**row))
    return out


def _validate(args):
    profile = CalibrationProfile.load(args.profile)
    print(
        json.dumps(
            {
                "name": profile.name,
                "target": profile.target,
                "model_version": profile.model_version,
                "parameters": len(profile.parameters),
                "fingerprint": profile.fingerprint(),
            },
            sort_keys=True,
        )
    )
    return 0


def _fit(args):
    base = CalibrationProfile.load(args.base)
    probes = _load_records(args.probes, ProbeSpec)
    measurements = _load_records(args.measurements, MeasurementRecord)
    fitted = fit_profile(
        base,
        probes,
        measurements,
        name=args.name,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    fitted.save(args.output)
    print(fitted.fingerprint())
    return 0


def _diff(args):
    before = CalibrationProfile.load(args.before)
    after = CalibrationProfile.load(args.after)
    if before.target != after.target or before.model_version != after.model_version:
        raise ValueError("profiles target different target/model versions")
    names = sorted(set(before.parameters) | set(after.parameters))
    changes = []
    for name in names:
        left = before.parameters.get(name)
        right = after.parameters.get(name)
        if left != right:
            changes.append(
                {
                    "parameter": name,
                    "before": None if left is None else left.value,
                    "after": None if right is None else right.value,
                    "provenance": None if right is None else right.provenance,
                    "measurements": (
                        [] if right is None else list(right.measurement_ids)
                    ),
                }
            )
    print(json.dumps(changes, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m allo.perf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate and fingerprint a profile"
    )
    validate.add_argument("profile")
    validate.set_defaults(run=_validate)

    fit = subparsers.add_parser("fit", help="fit isolated microprofile measurements")
    fit.add_argument("--base", required=True)
    fit.add_argument("--probes", required=True)
    fit.add_argument("--measurements", required=True)
    fit.add_argument("--name", required=True)
    fit.add_argument("--output", required=True)
    fit.set_defaults(run=_fit)

    diff = subparsers.add_parser("diff", help="show parameter changes between profiles")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.set_defaults(run=_diff)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
