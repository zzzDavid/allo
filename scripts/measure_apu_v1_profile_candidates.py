#!/usr/bin/env python3
"""Measure structural APUv1 profile candidates on a correctness-checked board.

The default runner uses the repository APUv1 vector runtime.  Compiler and
runner hooks are explicit arguments to the library entry point so tests and
alternative board harnesses can supply strict provenance-bearing records
without importing a proprietary SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from allo.pim.apu_v1_profile_search import (
    APUV1ProfileRowDecision,
    APUV1ProfileRowMaterialization,
    _ProfileProblem,
    _materialize,
    _score,
    derive_apu_v1_profile_plan_decisions,
    derive_apu_v1_profile_row_tiles,
)
from allo.pim.apu_v1_vector_program import APUV1PlanDecision
from allo.pim.apu_v1_vectorize import ContractionAnalysis, LogicalAxis, ValueAccess
from allo.pim.costs import apu_v1_cost
from allo.pim.schedule_search import InfeasibleSchedule
from allo.pim.targets import build_apu_v1_target


SCHEMA = "apu-v1-profile-candidate-measurements-v3"
MIN_MEASURED_SAMPLES = 5
DEFAULT_MEASURED_SAMPLES = 7
_PLATFORM_SCHEMA = "tenon-promotion-platform-v1"
_PLATFORM_TARGET = "apu_v1"
_PLATFORM_HARDWARE_FAMILY = "gemini-i"
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}\Z")
_COST_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}\Z")
_REPORT_FIELDS = frozenset(
    {
        "schema",
        "logical_shape",
        "protocol",
        "provenance",
        "domain",
        "candidates",
        "rejections",
        "promotion_eligibility",
        "report_fingerprint",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_identity",
        "candidate_identity_fingerprint",
        "decision",
        "composition",
        "model_prediction",
        "provenance",
        "waves",
        "warmup",
        "correlated_cycle_samples",
        "correctness",
        "measurement_fingerprint",
    }
)


class ProfileMeasurementError(ValueError):
    """Raised when measurement evidence cannot be emitted truthfully."""


@dataclass(frozen=True)
class ProfileCompileRequest:
    """Exact, name-free request passed to a candidate compiler hook."""

    wave_kind: str
    shape: tuple[int, int, int]
    emitted_source_fingerprint: str
    structural_source_fingerprint: str
    materialization_fingerprint: str
    plan_emitted_source_fingerprint: str
    target_fingerprint: str
    cost_fingerprint: str
    runtime_source_fingerprint: str
    platform_contract: Mapping[str, str] | None
    realization: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class CompiledProfileShard:
    """Compiler-hook result bound to the exact requested provenance."""

    wave_kind: str
    emitted_source_fingerprint: str
    materialization_fingerprint: str
    target_fingerprint: str
    cost_fingerprint: str
    runtime_source_fingerprint: str
    platform_contract: Mapping[str, str] | None
    payload: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class BoardExecution:
    """One correctness-checkable board execution returned by a runner hook."""

    sample_id: str
    cycles: int
    outputs: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    emitted_source_fingerprint: str
    materialization_fingerprint: str
    target_fingerprint: str
    cost_fingerprint: str
    runtime_source_fingerprint: str
    platform_contract: Mapping[str, str] | None


CompilerHook = Callable[[ProfileCompileRequest], CompiledProfileShard]
RunnerHook = Callable[[CompiledProfileShard, Mapping[str, np.ndarray]], BoardExecution]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(namespace: str, value: object) -> str:
    payload = _canonical_json({"namespace": namespace, "value": value})
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProfileMeasurementError(f"{field_name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ProfileMeasurementError(f"{field_name} must be a positive integer")
    return result


def _require_fingerprint(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
        raise ProfileMeasurementError(
            f"{field_name} must be a lowercase SHA-256 fingerprint"
        )
    return value


def _require_cost_fingerprint(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _COST_FINGERPRINT_RE.fullmatch(value) is None:
        raise ProfileMeasurementError(
            f"{field_name} must be an exact hexadecimal cost fingerprint"
        )
    return value


def _normalize_platform_contract(
    value: object, field_name: str
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProfileMeasurementError(f"{field_name} is not a platform contract")
    parts = tuple(value)
    if len(parts) != 4:
        raise ProfileMeasurementError(f"{field_name} is not a platform contract")
    schema, target, hardware_family, fingerprint = parts
    if (
        schema != _PLATFORM_SCHEMA
        or target != _PLATFORM_TARGET
        or hardware_family != _PLATFORM_HARDWARE_FAMILY
    ):
        raise ProfileMeasurementError(f"{field_name} targets the wrong platform")
    return {
        "schema": schema,
        "target": target,
        "hardware_family": hardware_family,
        "fingerprint": _require_fingerprint(fingerprint, f"{field_name}.fingerprint"),
    }


def _promotion_eligibility(
    platform_contract: Mapping[str, str] | None,
) -> dict[str, object]:
    reasons = ["built_board_binary_fingerprint_unavailable"]
    if platform_contract is None:
        reasons.append("hardware_sdk_firmware_platform_fingerprint_unavailable")
    return {"eligible": False, "reason_codes": reasons}


_PROMOTION_ELIGIBILITY = _promotion_eligibility(None)


def _profile_problem(analysis, target, cost):
    return _ProfileProblem(analysis, target, cost, None)


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field_name: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unsupported " + ", ".join(unknown))
        raise ProfileMeasurementError(
            f"{field_name} has invalid fields: {'; '.join(details)}"
        )


def _shape_manifest(shape: tuple[int, int, int]) -> dict[str, int]:
    rows, reduction, columns = shape
    return {"M": rows, "K": reduction, "N": columns}


def _decision_manifest(decision: APUV1PlanDecision) -> dict[str, object]:
    if not isinstance(decision, APUV1PlanDecision):
        raise ProfileMeasurementError("candidate decision is not an APUv1 plan")
    return {
        "temporal_strategy": decision.temporal_strategy,
        "reduction": {
            "kind": decision.reduction_kind,
            "group_size": decision.reduction_group_size,
            "tile_extent": decision.reduction_tile_extent,
        },
        "accumulator_block": decision.accumulator_block,
        "physical_iteration_shape": list(decision.physical_iteration_shape),
        "output_tile_shape": list(decision.output_tile_shape),
        "transfer_routes": [list(route) for route in decision.transfer_routes],
        "physical_fingerprint": _require_fingerprint(
            decision.physical_fingerprint, "decision.physical_fingerprint"
        ),
    }


def _structural_analysis(rows: int, reduction: int, columns: int):
    rows = _positive_int(rows, "M")
    reduction = _positive_int(reduction, "K")
    columns = _positive_int(columns, "N")
    row_axis = "parallel_output_0"
    column_axis = "parallel_output_1"
    reduction_axis = "reduction_0"
    axes = (
        LogicalAxis(row_axis, rows, 0, rows, 1, False, "%axis_0"),
        LogicalAxis(column_axis, columns, 0, columns, 1, False, "%axis_1"),
        LogicalAxis(reduction_axis, reduction, 0, reduction, 1, True, "%axis_2"),
    )
    return ContractionAnalysis(
        "structural_profile_measurement",
        axes,
        (row_axis, column_axis),
        (row_axis, column_axis),
        reduction_axis,
        ValueAccess(
            "operand_0",
            (row_axis, reduction_axis),
            "read",
            "ui16",
            (rows, reduction),
        ),
        ValueAccess(
            "operand_1",
            (reduction_axis, column_axis),
            "read",
            "ui16",
            (reduction, columns),
        ),
        ValueAccess(
            "output_0",
            (row_axis, column_axis),
            "read",
            "ui16",
            (rows, columns),
        ),
        ValueAccess(
            "output_0",
            (row_axis, column_axis),
            "write",
            "ui16",
            (rows, columns),
        ),
        "arith.muli",
        "arith.addi",
        "ui16",
    )


def _normalize_row_tiles(
    requested: Sequence[int] | None, derived: tuple[int, ...]
) -> tuple[int, ...]:
    if requested is None:
        return derived
    normalized = tuple(_positive_int(value, "row_tile") for value in requested)
    if len(set(normalized)) != len(normalized):
        raise ProfileMeasurementError("row-tile subset contains duplicates")
    unknown = sorted(set(normalized) - set(derived))
    if unknown:
        raise ProfileMeasurementError(
            f"row-tile subset is outside the structural domain: {unknown}"
        )
    return tuple(value for value in derived if value in set(normalized))


def _normalize_plan_fingerprints(
    requested: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if requested is None:
        return None
    normalized = tuple(
        _require_fingerprint(value, "physical_plan") for value in requested
    )
    if len(set(normalized)) != len(normalized):
        raise ProfileMeasurementError("physical-plan subset contains duplicates")
    return normalized


def _composition_manifest(
    materialization: APUV1ProfileRowMaterialization,
) -> dict[str, int]:
    partial = int(materialization.final_partial_shape is not None)
    full_waves = materialization.row_waves - partial
    columns = materialization.column_repetitions
    return {
        "row_waves": materialization.row_waves,
        "full_wave_count": full_waves,
        "partial_wave_count": partial,
        "column_repetitions": columns,
        "full_wave_executions": full_waves * columns,
        "partial_wave_executions": partial * columns,
        "total_wave_executions": materialization.row_waves * columns,
    }


def _candidate_identity(
    logical_shape: tuple[int, int, int],
    materialization: APUV1ProfileRowMaterialization,
) -> dict[str, object]:
    return {
        "logical_shape": _shape_manifest(logical_shape),
        "row_tile": materialization.decision.row_tile,
        "physical_plan": _decision_manifest(materialization.decision.plan),
        "full_wave_shape": _shape_manifest(
            (
                materialization.shard_shape.rows,
                materialization.shard_shape.reduction,
                materialization.shard_shape.columns,
            )
        ),
        "partial_wave_shape": (
            None
            if materialization.final_partial_shape is None
            else _shape_manifest(
                (
                    materialization.final_partial_shape.rows,
                    materialization.final_partial_shape.reduction,
                    materialization.final_partial_shape.columns,
                )
            )
        ),
        "partial_plan_fingerprint": materialization.final_plan_fingerprint,
        "composition": _composition_manifest(materialization),
    }


def _requested_identity(
    logical_shape: tuple[int, int, int], row_tile: int, decision: APUV1PlanDecision
) -> dict[str, object]:
    return {
        "logical_shape": _shape_manifest(logical_shape),
        "row_tile": row_tile,
        "physical_plan": _decision_manifest(decision),
    }


def _wave_request(
    materialization: APUV1ProfileRowMaterialization, wave_kind: str
) -> ProfileCompileRequest:
    if wave_kind == "full":
        realization = materialization.realization
        shape = (
            materialization.shard_shape.rows,
            materialization.shard_shape.reduction,
            materialization.shard_shape.columns,
        )
        emitted = materialization.emitted_source_fingerprint
        structural = materialization.structural_source_fingerprint
        plan_emitted = materialization.plan_emitted_source_fingerprint
    elif wave_kind == "partial" and materialization.final_realization is not None:
        realization = materialization.final_realization
        partial = materialization.final_partial_shape
        shape = (partial.rows, partial.reduction, partial.columns)
        emitted = materialization.final_emitted_source_fingerprint
        structural = materialization.final_structural_source_fingerprint
        plan_emitted = materialization.final_plan_emitted_source_fingerprint
    else:
        raise ProfileMeasurementError(f"candidate has no {wave_kind!r} wave")
    source = realization.device_source()
    actual_emitted = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual_emitted != emitted:
        raise ProfileMeasurementError(
            f"{wave_kind} emitted-source fingerprint changed before compilation"
        )
    materialization_fingerprint = _require_fingerprint(
        realization.promotion_materialization_fingerprint,
        f"{wave_kind}.materialization_fingerprint",
    )
    runtime_artifact = getattr(realization, "runtime_artifact", None)
    if runtime_artifact is None:
        raise ProfileMeasurementError(
            f"{wave_kind} realization has no complete runtime artifact"
        )
    try:
        runtime_artifact.assert_current(realization)
    except RuntimeError as error:
        raise ProfileMeasurementError(
            f"{wave_kind} runtime artifact is stale: {error}"
        ) from error
    runtime_source_fingerprint = _require_fingerprint(
        runtime_artifact.source_fingerprint,
        f"{wave_kind}.runtime_source_fingerprint",
    )
    platform_contract = _normalize_platform_contract(
        runtime_artifact.current_platform_fingerprint(),
        f"{wave_kind}.platform_contract",
    )
    return ProfileCompileRequest(
        wave_kind=wave_kind,
        shape=shape,
        emitted_source_fingerprint=_require_fingerprint(
            emitted, f"{wave_kind}.emitted_source_fingerprint"
        ),
        structural_source_fingerprint=_require_fingerprint(
            structural, f"{wave_kind}.structural_source_fingerprint"
        ),
        materialization_fingerprint=materialization_fingerprint,
        plan_emitted_source_fingerprint=_require_fingerprint(
            plan_emitted, f"{wave_kind}.plan_emitted_source_fingerprint"
        ),
        target_fingerprint=_require_fingerprint(
            materialization.target_provenance.fingerprint, "target_fingerprint"
        ),
        cost_fingerprint=_require_cost_fingerprint(
            materialization.cost_provenance.fingerprint, "cost_fingerprint"
        ),
        runtime_source_fingerprint=runtime_source_fingerprint,
        platform_contract=platform_contract,
        realization=realization,
    )


def default_compiler(request: ProfileCompileRequest) -> CompiledProfileShard:
    """Retain one exact realization for the SDK-backed board runner."""

    if not isinstance(request, ProfileCompileRequest):
        raise ProfileMeasurementError("compiler request is malformed")
    return CompiledProfileShard(
        request.wave_kind,
        request.emitted_source_fingerprint,
        request.materialization_fingerprint,
        request.target_fingerprint,
        request.cost_fingerprint,
        request.runtime_source_fingerprint,
        request.platform_contract,
        request.realization,
    )


def default_runner(
    compiled: CompiledProfileShard, inputs: Mapping[str, np.ndarray]
) -> BoardExecution:
    """Execute one exact realization through the installed APUv1 runtime."""

    from allo.pim.apu_v1_vector_runtime import run_apu_v1_vector

    arrays = {name: np.array(value, copy=True) for name, value in inputs.items()}
    wrapper = SimpleNamespace(realization=compiled.payload, realization_error=None)
    runtime_artifact = getattr(compiled.payload, "runtime_artifact", None)
    if runtime_artifact is None:
        raise ProfileMeasurementError("compiled shard has no complete runtime artifact")
    result = run_apu_v1_vector(
        wrapper,
        arrays,
        lab_name=runtime_artifact.lab_name,
    )
    cycles = _positive_int(result.cycles, "board cycles")
    outputs = result.extra.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ProfileMeasurementError("board result has no output mapping")
    runtime_source_fingerprint = _require_fingerprint(
        result.extra.get("source_fingerprint"),
        "board runtime_source_fingerprint",
    )
    platform_contract = _normalize_platform_contract(
        result.extra.get("promotion_platform_fingerprint"),
        "board platform_contract",
    )
    evidence = hashlib.sha256()
    evidence.update(str(result.stdout).encode("utf-8"))
    evidence.update(str(cycles).encode("ascii"))
    for name in sorted(outputs):
        value = np.asarray(outputs[name])
        evidence.update(name.encode("utf-8"))
        evidence.update(value.dtype.str.encode("ascii"))
        evidence.update(repr(value.shape).encode("ascii"))
        evidence.update(value.tobytes())
    return BoardExecution(
        evidence.hexdigest(),
        cycles,
        outputs,
        compiled.emitted_source_fingerprint,
        compiled.materialization_fingerprint,
        compiled.target_fingerprint,
        compiled.cost_fingerprint,
        runtime_source_fingerprint,
        platform_contract,
    )


def _validate_compiled(
    request: ProfileCompileRequest, compiled: object
) -> CompiledProfileShard:
    if not isinstance(compiled, CompiledProfileShard):
        raise ProfileMeasurementError(
            "compiler hook must return a CompiledProfileShard"
        )
    expected = (
        request.wave_kind,
        request.emitted_source_fingerprint,
        request.materialization_fingerprint,
        request.target_fingerprint,
        request.cost_fingerprint,
        request.runtime_source_fingerprint,
        request.platform_contract,
    )
    observed = (
        compiled.wave_kind,
        compiled.emitted_source_fingerprint,
        compiled.materialization_fingerprint,
        compiled.target_fingerprint,
        compiled.cost_fingerprint,
        compiled.runtime_source_fingerprint,
        compiled.platform_contract,
    )
    if observed != expected:
        raise ProfileMeasurementError(
            f"{request.wave_kind} compiler returned mixed provenance"
        )
    if compiled.payload is None:
        raise ProfileMeasurementError(
            f"{request.wave_kind} compiler returned no executable payload"
        )
    return compiled


def _wave_inputs(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    rows, reduction, columns = shape
    lhs = (
        ((np.arange(rows * reduction, dtype=np.uint64) * 5 + 1) % 7)
        .reshape(rows, reduction)
        .astype(np.uint16)
    )
    rhs = (
        ((np.arange(reduction * columns, dtype=np.uint64) * 3 + 2) % 5)
        .reshape(reduction, columns)
        .astype(np.uint16)
    )
    return {
        "operand_0": lhs,
        "operand_1": rhs,
        "output_0": np.zeros((rows, columns), dtype=np.uint16),
    }


def _expected_outputs(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    lhs = np.asarray(inputs["operand_0"], dtype=np.uint16)
    rhs = np.asarray(inputs["operand_1"], dtype=np.uint16)
    initial = np.asarray(inputs["output_0"], dtype=np.uint16)
    product = lhs.astype(np.uint64) @ rhs.astype(np.uint64)
    return {"output_0": (product + initial.astype(np.uint64)).astype(np.uint16)}


def _validate_execution(
    execution: object,
    compiled: CompiledProfileShard,
    expected_outputs: Mapping[str, np.ndarray],
    seen_sample_ids: set[str],
) -> dict[str, object]:
    if not isinstance(execution, BoardExecution):
        raise ProfileMeasurementError("runner hook must return a BoardExecution")
    sample_id = _require_fingerprint(execution.sample_id, "sample_id")
    if sample_id in seen_sample_ids:
        raise ProfileMeasurementError(f"duplicate board sample {sample_id}")
    seen_sample_ids.add(sample_id)
    expected_provenance = (
        compiled.emitted_source_fingerprint,
        compiled.materialization_fingerprint,
        compiled.target_fingerprint,
        compiled.cost_fingerprint,
        compiled.runtime_source_fingerprint,
        compiled.platform_contract,
    )
    observed_provenance = (
        execution.emitted_source_fingerprint,
        execution.materialization_fingerprint,
        execution.target_fingerprint,
        execution.cost_fingerprint,
        execution.runtime_source_fingerprint,
        execution.platform_contract,
    )
    if observed_provenance != expected_provenance:
        raise ProfileMeasurementError("board execution has mixed provenance")
    cycles = _positive_int(execution.cycles, "board cycles")
    if not isinstance(execution.outputs, Mapping):
        raise ProfileMeasurementError("board outputs must be a mapping")
    if set(execution.outputs) != set(expected_outputs):
        raise ProfileMeasurementError("board outputs do not match the oracle roles")
    for name, expected in expected_outputs.items():
        observed = np.asarray(execution.outputs[name])
        if observed.dtype != expected.dtype or observed.shape != expected.shape:
            raise ProfileMeasurementError(
                f"board output {name!r} has the wrong dtype or shape"
            )
        if not np.array_equal(observed, expected):
            raise ProfileMeasurementError(
                f"board output {name!r} failed bit-exact correctness"
            )
    return {"sample_id": sample_id, "cycles": cycles}


def _wave_evidence(request: ProfileCompileRequest) -> dict[str, object]:
    return {
        "kind": request.wave_kind,
        "shape": _shape_manifest(request.shape),
        "emitted_source_fingerprint": request.emitted_source_fingerprint,
        "structural_source_fingerprint": request.structural_source_fingerprint,
        "plan_emitted_source_fingerprint": request.plan_emitted_source_fingerprint,
        "materialization_fingerprint": request.materialization_fingerprint,
        "runtime_source_fingerprint": request.runtime_source_fingerprint,
        "platform_contract": request.platform_contract,
    }


def _run_wave(
    runner_hook: RunnerHook,
    compiled: CompiledProfileShard,
    inputs: Mapping[str, np.ndarray],
    expected_outputs: Mapping[str, np.ndarray],
    seen_sample_ids: set[str],
) -> dict[str, object]:
    try:
        execution = runner_hook(
            compiled,
            {name: np.array(value, copy=True) for name, value in inputs.items()},
        )
    except ProfileMeasurementError:
        raise
    except Exception as error:
        raise ProfileMeasurementError(
            f"{compiled.wave_kind} board runner failed: {error}"
        ) from error
    return _validate_execution(execution, compiled, expected_outputs, seen_sample_ids)


def _measure_materialization(
    logical_shape: tuple[int, int, int],
    materialization: APUV1ProfileRowMaterialization,
    model_prediction: Mapping[str, object],
    measured_samples: int,
    compiler_hook: CompilerHook,
    runner_hook: RunnerHook,
    seen_sample_ids: set[str],
) -> dict[str, object]:
    identity = _candidate_identity(logical_shape, materialization)
    identity_fingerprint = _fingerprint(
        "apu-v1-profile-candidate-structural-identity-v1", identity
    )
    full_request = _wave_request(materialization, "full")
    partial_request = (
        None
        if materialization.final_realization is None
        else _wave_request(materialization, "partial")
    )
    requests = tuple(
        request for request in (full_request, partial_request) if request is not None
    )
    if len({_canonical_json(request.platform_contract) for request in requests}) != 1:
        raise ProfileMeasurementError("candidate waves have mixed platform provenance")
    compiled = {}
    for request in requests:
        try:
            result = compiler_hook(request)
        except ProfileMeasurementError:
            raise
        except Exception as error:
            raise ProfileMeasurementError(
                f"{request.wave_kind} compiler hook failed: {error}"
            ) from error
        compiled[request.wave_kind] = _validate_compiled(request, result)
    inputs = {request.wave_kind: _wave_inputs(request.shape) for request in requests}
    expected = {
        wave_kind: _expected_outputs(wave_inputs)
        for wave_kind, wave_inputs in inputs.items()
    }

    warmup_full = _run_wave(
        runner_hook,
        compiled["full"],
        inputs["full"],
        expected["full"],
        seen_sample_ids,
    )
    warmup_partial = (
        None
        if partial_request is None
        else _run_wave(
            runner_hook,
            compiled["partial"],
            inputs["partial"],
            expected["partial"],
            seen_sample_ids,
        )
    )
    composition = _composition_manifest(materialization)
    samples = []
    for index in range(measured_samples):
        full = _run_wave(
            runner_hook,
            compiled["full"],
            inputs["full"],
            expected["full"],
            seen_sample_ids,
        )
        partial = (
            None
            if partial_request is None
            else _run_wave(
                runner_hook,
                compiled["partial"],
                inputs["partial"],
                expected["partial"],
                seen_sample_ids,
            )
        )
        composed_cycles = (
            int(full["cycles"]) * composition["full_wave_executions"]
            + (0 if partial is None else int(partial["cycles"]))
            * composition["partial_wave_executions"]
        )
        samples.append(
            {
                "index": index,
                "full_wave": full,
                "partial_wave": partial,
                "composed_cycles": composed_cycles,
            }
        )

    body = {
        "candidate_identity": identity,
        "candidate_identity_fingerprint": identity_fingerprint,
        "decision": _decision_manifest(materialization.decision.plan),
        "composition": composition,
        "model_prediction": dict(model_prediction),
        "provenance": {
            "profile_materialization_fingerprint": _require_fingerprint(
                materialization.promotion_materialization_fingerprint,
                "profile_materialization_fingerprint",
            ),
            "target_fingerprint": full_request.target_fingerprint,
            "cost_fingerprint": full_request.cost_fingerprint,
            "platform_contract": full_request.platform_contract,
        },
        "waves": {
            "full": _wave_evidence(full_request),
            "partial": (
                None if partial_request is None else _wave_evidence(partial_request)
            ),
        },
        "warmup": {
            "count": 1,
            "discarded": True,
            "full_wave": warmup_full,
            "partial_wave": warmup_partial,
        },
        "correlated_cycle_samples": samples,
        "correctness": {
            "status": "pass",
            "oracle": "bit-exact structural uint16 contraction",
            "checked_board_executions": (measured_samples + 1) * len(requests),
        },
    }
    return {
        **body,
        "measurement_fingerprint": _fingerprint(
            "apu-v1-profile-candidate-measurement-v1", body
        ),
    }


def measure_apu_v1_profile_candidates(
    rows: int,
    reduction: int,
    columns: int,
    *,
    row_tiles: Sequence[int] | None = None,
    physical_plan_fingerprints: Sequence[str] | None = None,
    measured_samples: int = DEFAULT_MEASURED_SAMPLES,
    compiler_hook: CompilerHook = default_compiler,
    runner_hook: RunnerHook = default_runner,
    target: object | None = None,
    cost: object = apu_v1_cost,
) -> dict[str, object]:
    """Measure every selected materializable structural profile candidate."""

    measured_samples = _positive_int(measured_samples, "measured_samples")
    if measured_samples < MIN_MEASURED_SAMPLES:
        raise ProfileMeasurementError(
            f"measured_samples must be at least {MIN_MEASURED_SAMPLES}"
        )
    logical_shape = (
        _positive_int(rows, "M"),
        _positive_int(reduction, "K"),
        _positive_int(columns, "N"),
    )
    analysis = _structural_analysis(*logical_shape)
    target = build_apu_v1_target() if target is None else target
    try:
        derived_rows = derive_apu_v1_profile_row_tiles(analysis, target)
        selected_rows = _normalize_row_tiles(row_tiles, derived_rows)
        selected_plans = _normalize_plan_fingerprints(physical_plan_fingerprints)
        problem = _profile_problem(analysis, target, cost)
    except ProfileMeasurementError:
        raise
    except Exception as error:
        raise ProfileMeasurementError(
            f"cannot derive the structural candidate domain: {error}"
        ) from error

    plan_filter = None if selected_plans is None else set(selected_plans)
    matched_plan_fingerprints: set[str] = set()
    materializations = []
    rejections = []
    requested_identity_fingerprints: set[str] = set()
    decisions_by_row = {
        row_tile: derive_apu_v1_profile_plan_decisions(
            analysis,
            target,
            row_tile=row_tile,
        )
        for row_tile in derived_rows
    }
    full_domain_candidate_count = sum(
        len(decisions) for decisions in decisions_by_row.values()
    )
    selected_candidate_count = 0
    for row_tile in selected_rows:
        decisions = decisions_by_row[row_tile]
        expected_decisions = tuple(
            entry.decision for entry in problem.entries(row_tile)
        )
        if decisions != expected_decisions:
            raise ProfileMeasurementError(
                "candidate derivation and materialization domains have mixed provenance"
            )
        for decision in decisions:
            if (
                plan_filter is not None
                and decision.physical_fingerprint not in plan_filter
            ):
                continue
            selected_candidate_count += 1
            matched_plan_fingerprints.add(decision.physical_fingerprint)
            requested_identity = _requested_identity(logical_shape, row_tile, decision)
            requested_fingerprint = _fingerprint(
                "apu-v1-profile-candidate-request-v1", requested_identity
            )
            if requested_fingerprint in requested_identity_fingerprints:
                raise ProfileMeasurementError("candidate domain contains duplicates")
            requested_identity_fingerprints.add(requested_fingerprint)
            try:
                materialization = _materialize(
                    problem, APUV1ProfileRowDecision(row_tile, decision)
                )
            except InfeasibleSchedule as error:
                rejections.append(
                    {
                        "requested_identity": requested_identity,
                        "requested_identity_fingerprint": requested_fingerprint,
                        "stage": "materialization",
                        "reason": str(error),
                    }
                )
                continue
            materializations.append(materialization)

    if selected_plans is not None:
        unknown = sorted(set(selected_plans) - matched_plan_fingerprints)
        if unknown:
            raise ProfileMeasurementError(
                f"physical-plan subset is outside the selected domain: {unknown}"
            )
    if selected_candidate_count == 0:
        raise ProfileMeasurementError("candidate subset is empty")
    if not materializations:
        raise ProfileMeasurementError("candidate subset has no materializable plans")

    seen_candidate_identities: set[str] = set()
    seen_sample_ids: set[str] = set()
    candidates = []
    for materialization in materializations:
        score = _score(problem, materialization)
        model_prediction = {
            "shard_cycles": score.shard_cycles,
            "final_shard_cycles": score.final_shard_cycles,
            "composed_cycles": score.composed_cycles,
            "objective_domain": score.objective_domain.manifest(),
        }
        candidate = _measure_materialization(
            logical_shape,
            materialization,
            model_prediction,
            measured_samples,
            compiler_hook,
            runner_hook,
            seen_sample_ids,
        )
        identity_fingerprint = str(candidate["candidate_identity_fingerprint"])
        if identity_fingerprint in seen_candidate_identities:
            raise ProfileMeasurementError(
                "materialized candidate domain has duplicates"
            )
        seen_candidate_identities.add(identity_fingerprint)
        candidates.append(candidate)

    target_fingerprints = {
        candidate["provenance"]["target_fingerprint"] for candidate in candidates
    }
    cost_fingerprints = {
        candidate["provenance"]["cost_fingerprint"] for candidate in candidates
    }
    platform_contracts = {
        _canonical_json(candidate["provenance"]["platform_contract"])
        for candidate in candidates
    }
    if (
        len(target_fingerprints) != 1
        or len(cost_fingerprints) != 1
        or len(platform_contracts) != 1
    ):
        raise ProfileMeasurementError("candidate measurements have mixed provenance")
    platform_contract = candidates[0]["provenance"]["platform_contract"]
    body = {
        "schema": SCHEMA,
        "logical_shape": _shape_manifest(logical_shape),
        "protocol": {
            "warmup_rounds": 1,
            "discarded_warmup_rounds": 1,
            "measured_samples_per_candidate": measured_samples,
            "minimum_measured_samples": MIN_MEASURED_SAMPLES,
            "sample_correlation": "full and partial wave cycles share a round index",
            "correctness": "every warmup and measured board execution is bit-exact",
        },
        "provenance": {
            "target_fingerprint": next(iter(target_fingerprints)),
            "cost_fingerprint": next(iter(cost_fingerprints)),
            "platform_contract": platform_contract,
        },
        "domain": {
            "derived_row_tiles": list(derived_rows),
            "selected_row_tiles": list(selected_rows),
            "selected_physical_plan_fingerprints": (
                None if selected_plans is None else list(selected_plans)
            ),
            "row_tile_domain_complete": selected_rows == derived_rows,
            "physical_plan_domain_complete": selected_plans is None,
            "full_domain_candidate_count": full_domain_candidate_count,
            "selected_candidate_count": selected_candidate_count,
            "materialized_candidate_count": len(materializations),
            "rejected_candidate_count": len(rejections),
            "measured_candidate_count": len(candidates),
            "candidate_domain_complete": (
                selected_rows == derived_rows
                and selected_plans is None
                and not rejections
                and len(candidates) == full_domain_candidate_count
            ),
        },
        "candidates": candidates,
        "rejections": rejections,
        "promotion_eligibility": _promotion_eligibility(platform_contract),
    }
    return {
        **body,
        "report_fingerprint": _fingerprint(
            "apu-v1-profile-candidate-measurement-report-v1", body
        ),
    }


def _report_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileMeasurementError(f"{field_name} must be an object")
    return value


def _report_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProfileMeasurementError(f"{field_name} must be an array")
    return value


def _validated_shape(value: object, field_name: str) -> tuple[int, int, int]:
    shape = _report_mapping(value, field_name)
    _require_exact_fields(shape, frozenset({"M", "K", "N"}), field_name)
    return tuple(
        _positive_int(shape[axis], f"{field_name}.{axis}") for axis in ("M", "K", "N")
    )


def _validated_cycle_record(
    value: object,
    field_name: str,
    seen_sample_ids: set[str],
) -> dict[str, object]:
    record = _report_mapping(value, field_name)
    _require_exact_fields(record, frozenset({"sample_id", "cycles"}), field_name)
    sample_id = _require_fingerprint(record["sample_id"], f"{field_name}.sample_id")
    if sample_id in seen_sample_ids:
        raise ProfileMeasurementError(f"{field_name} repeats a board sample ID")
    seen_sample_ids.add(sample_id)
    return {
        "sample_id": sample_id,
        "cycles": _positive_int(record["cycles"], f"{field_name}.cycles"),
    }


def validate_measurement_report(
    report: Mapping[str, Any],
    *,
    target: object | None = None,
    cost: object = apu_v1_cost,
) -> None:
    """Rebuild the domain/model and validate every nested measurement record."""

    if not isinstance(report, Mapping):
        raise ProfileMeasurementError("measurement report must be an object")
    _require_exact_fields(report, _REPORT_FIELDS, "report")
    if report.get("schema") != SCHEMA:
        raise ProfileMeasurementError(f"report.schema must be {SCHEMA!r}")

    logical_shape = _validated_shape(
        report.get("logical_shape"), "report.logical_shape"
    )
    protocol = _report_mapping(report.get("protocol"), "report.protocol")
    _require_exact_fields(
        protocol,
        frozenset(
            {
                "warmup_rounds",
                "discarded_warmup_rounds",
                "measured_samples_per_candidate",
                "minimum_measured_samples",
                "sample_correlation",
                "correctness",
            }
        ),
        "report.protocol",
    )
    measured_samples = _positive_int(
        protocol["measured_samples_per_candidate"],
        "report.protocol.measured_samples_per_candidate",
    )
    expected_protocol = {
        "warmup_rounds": 1,
        "discarded_warmup_rounds": 1,
        "measured_samples_per_candidate": measured_samples,
        "minimum_measured_samples": MIN_MEASURED_SAMPLES,
        "sample_correlation": "full and partial wave cycles share a round index",
        "correctness": "every warmup and measured board execution is bit-exact",
    }
    if measured_samples < MIN_MEASURED_SAMPLES or dict(protocol) != expected_protocol:
        raise ProfileMeasurementError("report protocol is inconsistent")

    domain = _report_mapping(report.get("domain"), "report.domain")
    derived_rows_raw = _report_sequence(
        domain.get("derived_row_tiles"), "report.domain.derived_row_tiles"
    )
    selected_rows_raw = _report_sequence(
        domain.get("selected_row_tiles"), "report.domain.selected_row_tiles"
    )
    selected_plans_raw = domain.get("selected_physical_plan_fingerprints")
    selected_plans = (
        None
        if selected_plans_raw is None
        else _normalize_plan_fingerprints(
            _report_sequence(
                selected_plans_raw,
                "report.domain.selected_physical_plan_fingerprints",
            )
        )
    )

    analysis = _structural_analysis(*logical_shape)
    target = build_apu_v1_target() if target is None else target
    try:
        derived_rows = derive_apu_v1_profile_row_tiles(analysis, target)
        selected_rows = _normalize_row_tiles(selected_rows_raw, derived_rows)
        problem = _profile_problem(analysis, target, cost)
    except ProfileMeasurementError:
        raise
    except Exception as error:
        raise ProfileMeasurementError(
            f"cannot rebuild the structural candidate domain: {error}"
        ) from error
    observed_derived_rows = tuple(
        _positive_int(value, "report.domain.derived_row_tiles")
        for value in derived_rows_raw
    )
    if observed_derived_rows != derived_rows:
        raise ProfileMeasurementError("report has a stale derived row-tile domain")

    plan_filter = None if selected_plans is None else set(selected_plans)
    matched_plans: set[str] = set()
    expected_materializations = []
    expected_rejections = []
    selected_candidate_count = 0
    decisions_by_row = {
        row_tile: derive_apu_v1_profile_plan_decisions(
            analysis,
            target,
            row_tile=row_tile,
        )
        for row_tile in derived_rows
    }
    for row_tile in selected_rows:
        decisions = decisions_by_row[row_tile]
        if decisions != tuple(entry.decision for entry in problem.entries(row_tile)):
            raise ProfileMeasurementError(
                "rebuilt derivation and materialization domains have mixed provenance"
            )
        for decision in decisions:
            if (
                plan_filter is not None
                and decision.physical_fingerprint not in plan_filter
            ):
                continue
            selected_candidate_count += 1
            matched_plans.add(decision.physical_fingerprint)
            requested_identity = _requested_identity(logical_shape, row_tile, decision)
            requested_fingerprint = _fingerprint(
                "apu-v1-profile-candidate-request-v1", requested_identity
            )
            try:
                materialization = _materialize(
                    problem,
                    APUV1ProfileRowDecision(row_tile, decision),
                )
            except InfeasibleSchedule as error:
                expected_rejections.append(
                    {
                        "requested_identity": requested_identity,
                        "requested_identity_fingerprint": requested_fingerprint,
                        "stage": "materialization",
                        "reason": str(error),
                    }
                )
                continue
            expected_materializations.append(materialization)
    if selected_plans is not None and set(selected_plans) != matched_plans:
        raise ProfileMeasurementError("report selects plans outside the rebuilt domain")

    rejections = list(_report_sequence(report.get("rejections"), "report.rejections"))
    if rejections != expected_rejections:
        raise ProfileMeasurementError(
            "report rejections do not match rebuilt candidates"
        )
    candidates = _report_sequence(report.get("candidates"), "report.candidates")
    if len(candidates) != len(expected_materializations) or not candidates:
        raise ProfileMeasurementError(
            "report candidates do not cover every rebuilt materialization"
        )

    full_domain_candidate_count = sum(
        len(decisions) for decisions in decisions_by_row.values()
    )
    expected_domain = {
        "derived_row_tiles": list(derived_rows),
        "selected_row_tiles": list(selected_rows),
        "selected_physical_plan_fingerprints": (
            None if selected_plans is None else list(selected_plans)
        ),
        "row_tile_domain_complete": selected_rows == derived_rows,
        "physical_plan_domain_complete": selected_plans is None,
        "full_domain_candidate_count": full_domain_candidate_count,
        "selected_candidate_count": selected_candidate_count,
        "materialized_candidate_count": len(expected_materializations),
        "rejected_candidate_count": len(expected_rejections),
        "measured_candidate_count": len(candidates),
        "candidate_domain_complete": (
            selected_rows == derived_rows
            and selected_plans is None
            and not expected_rejections
            and len(candidates) == full_domain_candidate_count
        ),
    }
    if dict(domain) != expected_domain:
        raise ProfileMeasurementError("report candidate-domain accounting is false")

    seen_identities: set[str] = set()
    seen_measurements: set[str] = set()
    seen_sample_ids: set[str] = set()
    expected_target_fingerprints: set[str] = set()
    expected_cost_fingerprints: set[str] = set()
    expected_platform_contracts: dict[str, Mapping[str, str] | None] = {}
    for index, (candidate_raw, materialization) in enumerate(
        zip(candidates, expected_materializations)
    ):
        field_name = f"report.candidates[{index}]"
        candidate = _report_mapping(candidate_raw, field_name)
        _require_exact_fields(candidate, _CANDIDATE_FIELDS, field_name)
        expected_identity = _candidate_identity(logical_shape, materialization)
        if candidate.get("candidate_identity") != expected_identity:
            raise ProfileMeasurementError(f"{field_name} has a stale physical identity")
        identity = _require_fingerprint(
            candidate.get("candidate_identity_fingerprint"),
            f"{field_name}.candidate_identity_fingerprint",
        )
        if identity != _fingerprint(
            "apu-v1-profile-candidate-structural-identity-v1", expected_identity
        ):
            raise ProfileMeasurementError(
                f"{field_name} has a false structural identity"
            )
        if identity in seen_identities:
            raise ProfileMeasurementError(
                "report contains duplicate candidate identities"
            )
        seen_identities.add(identity)

        expected_decision = _decision_manifest(materialization.decision.plan)
        expected_composition = _composition_manifest(materialization)
        if candidate.get("decision") != expected_decision:
            raise ProfileMeasurementError(f"{field_name} decision is stale")
        if candidate.get("composition") != expected_composition:
            raise ProfileMeasurementError(f"{field_name} composition is stale")
        score = _score(problem, materialization)
        expected_prediction = {
            "shard_cycles": score.shard_cycles,
            "final_shard_cycles": score.final_shard_cycles,
            "composed_cycles": score.composed_cycles,
            "objective_domain": score.objective_domain.manifest(),
        }
        if candidate.get("model_prediction") != expected_prediction:
            raise ProfileMeasurementError(f"{field_name} model prediction is stale")

        full_request = _wave_request(materialization, "full")
        partial_request = (
            None
            if materialization.final_realization is None
            else _wave_request(materialization, "partial")
        )
        expected_waves = {
            "full": _wave_evidence(full_request),
            "partial": (
                None if partial_request is None else _wave_evidence(partial_request)
            ),
        }
        if candidate.get("waves") != expected_waves:
            raise ProfileMeasurementError(f"{field_name} wave evidence is stale")
        expected_provenance = {
            "profile_materialization_fingerprint": _require_fingerprint(
                materialization.promotion_materialization_fingerprint,
                "profile_materialization_fingerprint",
            ),
            "target_fingerprint": full_request.target_fingerprint,
            "cost_fingerprint": full_request.cost_fingerprint,
            "platform_contract": full_request.platform_contract,
        }
        if candidate.get("provenance") != expected_provenance:
            raise ProfileMeasurementError(f"{field_name} provenance is stale")
        expected_target_fingerprints.add(full_request.target_fingerprint)
        expected_cost_fingerprints.add(full_request.cost_fingerprint)
        expected_platform_contracts[_canonical_json(full_request.platform_contract)] = (
            full_request.platform_contract
        )

        warmup = _report_mapping(candidate.get("warmup"), f"{field_name}.warmup")
        _require_exact_fields(
            warmup,
            frozenset({"count", "discarded", "full_wave", "partial_wave"}),
            f"{field_name}.warmup",
        )
        if warmup["count"] != 1 or warmup["discarded"] is not True:
            raise ProfileMeasurementError(f"{field_name} warmup protocol is false")
        _validated_cycle_record(
            warmup["full_wave"],
            f"{field_name}.warmup.full_wave",
            seen_sample_ids,
        )
        if partial_request is None:
            if warmup["partial_wave"] is not None:
                raise ProfileMeasurementError(
                    f"{field_name} has a false partial warmup"
                )
        else:
            _validated_cycle_record(
                warmup["partial_wave"],
                f"{field_name}.warmup.partial_wave",
                seen_sample_ids,
            )

        samples = _report_sequence(
            candidate.get("correlated_cycle_samples"),
            f"{field_name}.correlated_cycle_samples",
        )
        if len(samples) != measured_samples:
            raise ProfileMeasurementError(f"{field_name} has an invalid sample count")
        for sample_index, sample_raw in enumerate(samples):
            sample_field = f"{field_name}.correlated_cycle_samples[{sample_index}]"
            sample = _report_mapping(sample_raw, sample_field)
            _require_exact_fields(
                sample,
                frozenset({"index", "full_wave", "partial_wave", "composed_cycles"}),
                sample_field,
            )
            if sample["index"] != sample_index:
                raise ProfileMeasurementError(f"{sample_field} has a false round index")
            full = _validated_cycle_record(
                sample["full_wave"], f"{sample_field}.full_wave", seen_sample_ids
            )
            if partial_request is None:
                if sample["partial_wave"] is not None:
                    raise ProfileMeasurementError(
                        f"{sample_field} has a false partial wave"
                    )
                partial_cycles = 0
            else:
                partial = _validated_cycle_record(
                    sample["partial_wave"],
                    f"{sample_field}.partial_wave",
                    seen_sample_ids,
                )
                partial_cycles = int(partial["cycles"])
            expected_cycles = (
                int(full["cycles"]) * expected_composition["full_wave_executions"]
                + partial_cycles * expected_composition["partial_wave_executions"]
            )
            if sample["composed_cycles"] != expected_cycles:
                raise ProfileMeasurementError(
                    f"{sample_field} has false composed cycles"
                )

        expected_correctness = {
            "status": "pass",
            "oracle": "bit-exact structural uint16 contraction",
            "checked_board_executions": (measured_samples + 1)
            * (1 + int(partial_request is not None)),
        }
        if candidate.get("correctness") != expected_correctness:
            raise ProfileMeasurementError(f"{field_name} correctness evidence is false")
        measurement = _require_fingerprint(
            candidate.get("measurement_fingerprint"),
            f"{field_name}.measurement_fingerprint",
        )
        candidate_body = {
            key: value
            for key, value in candidate.items()
            if key != "measurement_fingerprint"
        }
        if measurement != _fingerprint(
            "apu-v1-profile-candidate-measurement-v1", candidate_body
        ):
            raise ProfileMeasurementError(
                f"{field_name} has a false measurement fingerprint"
            )
        if measurement in seen_measurements:
            raise ProfileMeasurementError("report contains duplicate measurements")
        seen_measurements.add(measurement)

    provenance = _report_mapping(report.get("provenance"), "report.provenance")
    if (
        len(expected_target_fingerprints) != 1
        or len(expected_cost_fingerprints) != 1
        or len(expected_platform_contracts) != 1
    ):
        raise ProfileMeasurementError("rebuilt candidates have mixed provenance")
    platform_contract = next(iter(expected_platform_contracts.values()))
    expected_report_provenance = {
        "target_fingerprint": next(iter(expected_target_fingerprints)),
        "cost_fingerprint": next(iter(expected_cost_fingerprints)),
        "platform_contract": platform_contract,
    }
    if dict(provenance) != expected_report_provenance:
        raise ProfileMeasurementError("report provenance is stale")
    if report.get("promotion_eligibility") != _promotion_eligibility(platform_contract):
        raise ProfileMeasurementError("report promotion eligibility is malformed")

    report_fingerprint = _require_fingerprint(
        report.get("report_fingerprint"), "report.report_fingerprint"
    )
    report_body = {
        key: value for key, value in report.items() if key != "report_fingerprint"
    }
    if report_fingerprint != _fingerprint(
        "apu-v1-profile-candidate-measurement-report-v1", report_body
    ):
        raise ProfileMeasurementError("report has a false report fingerprint")


def report_to_json(report: Mapping[str, Any]) -> str:
    """Serialize one validated measurement report deterministically."""

    validate_measurement_report(report)
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_measurement_report(
    path: Path, report: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    """Write strict JSON, refusing replacement unless explicitly requested."""

    path = Path(path)
    rendered = report_to_json(report)
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8") as output:
            output.write(rendered)
    except FileExistsError as error:
        raise ProfileMeasurementError(
            f"output already exists: {path}; pass --overwrite to replace it"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure structural APUv1 profile candidates on the board."
    )
    parser.add_argument("M", type=int, help="logical output-row extent")
    parser.add_argument("K", type=int, help="logical reduction extent")
    parser.add_argument("N", type=int, help="logical output-column extent")
    parser.add_argument(
        "--row-tile",
        type=int,
        action="append",
        dest="row_tiles",
        help="measure only this derived row tile; repeat for a subset",
    )
    parser.add_argument(
        "--physical-plan",
        action="append",
        dest="physical_plans",
        help="measure only this physical plan fingerprint; repeat for a subset",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_MEASURED_SAMPLES,
        help=f"measured samples per candidate (minimum {MIN_MEASURED_SAMPLES})",
    )
    parser.add_argument("--output", type=Path, required=True, help="strict JSON output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = measure_apu_v1_profile_candidates(
            args.M,
            args.K,
            args.N,
            row_tiles=args.row_tiles,
            physical_plan_fingerprints=args.physical_plans,
            measured_samples=args.samples,
        )
        write_measurement_report(args.output, report, overwrite=args.overwrite)
    except (ProfileMeasurementError, OSError, RuntimeError) as error:
        print(f"FAIL APUv1 profile candidate measurement: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
