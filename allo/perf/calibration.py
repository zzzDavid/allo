# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, agent-editable calibration artifacts.

Raw measurements are immutable records. ``fit_profile`` produces a new
profile overlay by robustly fitting probes that isolate one named parameter.
The analytical formulas themselves remain separate in ``TimingModel``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class ParameterValue:
    """One calibrated scalar and its provenance."""

    value: int | float
    unit: str = "cycles"
    lower: int | float | None = None
    upper: int | float | None = None
    provenance: str = "assumption"
    measurement_ids: tuple[str, ...] = ()
    valid_domain: Mapping[str, object] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError(f"parameter value must be finite, got {self.value!r}")
        if self.lower is not None and self.lower > self.value:
            raise ValueError("parameter lower bound exceeds its value")
        if self.upper is not None and self.upper < self.value:
            raise ValueError("parameter upper bound is below its value")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("parameter lower bound exceeds upper bound")


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable parameter overlay for one target/model version."""

    name: str
    target: str
    model_version: str
    parameters: Mapping[str, ParameterValue]
    target_fingerprint: str = ""
    parent: str | None = None
    created_at: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def get(self, name: str) -> ParameterValue:
        try:
            return self.parameters[name]
        except KeyError as exc:
            raise KeyError(
                f"calibration profile {self.name!r} has no parameter {name!r}; "
                f"available: {sorted(self.parameters)}"
            ) from exc

    def overlay(
        self,
        name: str,
        updates: Mapping[str, ParameterValue],
        **metadata,
    ) -> "CalibrationProfile":
        parameters = dict(self.parameters)
        unknown = set(updates) - set(parameters)
        if unknown:
            raise KeyError(
                f"calibration overlay introduces unknown parameters {sorted(unknown)}"
            )
        parameters.update(updates)
        return replace(
            self,
            name=name,
            parameters=parameters,
            parent=self.name,
            **metadata,
        )

    def fingerprint(self) -> str:
        payload = self.to_dict()
        payload.pop("created_at", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "name": self.name,
            "target": self.target,
            "model_version": self.model_version,
            "target_fingerprint": self.target_fingerprint,
            "parent": self.parent,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "parameters": {
                name: asdict(value) for name, value in sorted(self.parameters.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalibrationProfile":
        if data.get("schema_version", 1) != 1:
            raise ValueError(
                f"unsupported calibration schema {data.get('schema_version')!r}"
            )
        parameters = {
            name: ParameterValue(
                **{
                    **value,
                    "measurement_ids": tuple(value.get("measurement_ids", ())),
                }
            )
            for name, value in dict(data["parameters"]).items()
        }
        return cls(
            name=str(data["name"]),
            target=str(data["target"]),
            model_version=str(data["model_version"]),
            parameters=parameters,
            target_fingerprint=str(data.get("target_fingerprint", "")),
            parent=data.get("parent"),
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class ProbeSpec:
    """A microbenchmark that isolates one model parameter.

    For each observation, ``parameter = (cycles - fixed_cycles) / normalizer``.
    More complex multi-parameter fitting can be added without changing the raw
    measurement schema.
    """

    id: str
    parameter: str
    description: str = ""
    warmup_runs: int = 3
    measured_runs: int = 20


@dataclass(frozen=True)
class MeasurementRecord:
    """Immutable result of running one microbenchmark configuration."""

    id: str
    probe_id: str
    target: str
    target_fingerprint: str
    cycles_samples: tuple[int, ...]
    normalizer: float = 1.0
    fixed_cycles: float = 0.0
    inputs: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, object] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.id or not self.probe_id:
            raise ValueError("measurement id and probe_id must be non-empty")
        if not self.cycles_samples:
            raise ValueError("measurement must contain at least one cycle sample")
        if min(self.cycles_samples) < 0:
            raise ValueError("cycle samples must be non-negative")
        if self.normalizer <= 0:
            raise ValueError("measurement normalizer must be positive")

    def parameter_samples(self) -> tuple[float, ...]:
        return tuple(
            (sample - self.fixed_cycles) / self.normalizer
            for sample in self.cycles_samples
        )


def fit_profile(
    base: CalibrationProfile,
    probes: Iterable[ProbeSpec],
    measurements: Iterable[MeasurementRecord],
    *,
    name: str,
    created_at: str = "",
) -> CalibrationProfile:
    """Fit direct/isolation probes into a new immutable profile overlay.

    The median is robust to timing outliers. The 10th/90th empirical
    quantiles form a deliberately transparent uncertainty interval.
    """

    probe_by_id = {probe.id: probe for probe in probes}
    samples_by_parameter: dict[str, list[float]] = {}
    ids_by_parameter: dict[str, list[str]] = {}
    fingerprints = set()
    for record in measurements:
        if record.target != base.target:
            raise ValueError(
                f"measurement {record.id!r} targets {record.target!r}, "
                f"expected {base.target!r}"
            )
        try:
            probe = probe_by_id[record.probe_id]
        except KeyError as exc:
            raise KeyError(
                f"measurement {record.id!r} references unknown probe {record.probe_id!r}"
            ) from exc
        if probe.parameter not in base.parameters:
            raise KeyError(
                f"probe {probe.id!r} fits unknown parameter {probe.parameter!r}"
            )
        samples_by_parameter.setdefault(probe.parameter, []).extend(
            record.parameter_samples()
        )
        ids_by_parameter.setdefault(probe.parameter, []).append(record.id)
        if record.target_fingerprint:
            fingerprints.add(record.target_fingerprint)

    if len(fingerprints) > 1:
        raise ValueError(
            "cannot combine measurements from different target fingerprints: "
            f"{sorted(fingerprints)}"
        )

    updates: dict[str, ParameterValue] = {}
    for parameter, samples in samples_by_parameter.items():
        ordered = sorted(samples)
        n = len(ordered)
        lo = ordered[max(0, math.floor(0.10 * (n - 1)))]
        hi = ordered[min(n - 1, math.ceil(0.90 * (n - 1)))]
        value = statistics.median(ordered)
        prior = base.get(parameter)
        updates[parameter] = ParameterValue(
            value=value,
            unit=prior.unit,
            lower=min(lo, value),
            upper=max(hi, value),
            provenance="measured",
            measurement_ids=tuple(ids_by_parameter[parameter]),
            valid_domain=prior.valid_domain,
            note=f"robust median of {n} microprofile samples",
        )

    fingerprint = next(iter(fingerprints), base.target_fingerprint)
    return base.overlay(
        name,
        updates,
        target_fingerprint=fingerprint,
        created_at=created_at,
    )
