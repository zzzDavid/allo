# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests for persistent-GEMM candidate measurement tooling."""

import copy
import importlib.util
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import allo
import numpy as np
import pytest
from allo.ir.types import uint16
from allo.pim.apu_g2_contraction import plan_apu_g2_rank_n_contraction
from allo.pim.apu_g2_vector_program import (
    APUG2PersistentGemmSchedule,
    _materialize_apu_g2_persistent_gemm_schedule,
)
from allo.pim.costs.apu_g2 import estimate_apu_g2_u16_gemm_wall_us
from allo.pim.targets import build_apu_g2_target
from allo.spmw_codegen import RunResult


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "measure_apu_g2_persistent_candidates.py"
SPEC = importlib.util.spec_from_file_location(
    "measure_apu_g2_persistent_candidates", SCRIPT
)
MEASURER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MEASURER
SPEC.loader.exec_module(MEASURER)


def structural_gemm(A: uint16[2, 3], B: uint16[3, 4], output: uint16[2, 4]):
    for row, column in allo.grid(2, 4):
        for depth in allo.reduction(3):
            output[row, column] += A[row, depth] * B[depth, column]


def renamed_structural_gemm(
    left: uint16[2, 3], right: uint16[3, 4], result: uint16[2, 4]
):
    for outer, inner in allo.grid(2, 4):
        for contracted in allo.reduction(3):
            result[outer, inner] += left[outer, contracted] * right[contracted, inner]


def _fake_runner(provenance, *, fault=None):
    calls = []
    invocation_counts = defaultdict(int)

    def runner(lhs, rhs, accumulator, *, alpha, beta, batch_columns):
        invocation = invocation_counts[batch_columns]
        invocation_counts[batch_columns] += 1
        calls.append(
            {
                "batch_columns": batch_columns,
                "invocation": invocation,
                "alpha": alpha,
                "beta": beta,
            }
        )
        observed = (
            np.uint64(alpha) * (lhs.astype(np.uint64) @ rhs.astype(np.uint64))
            + np.uint64(beta) * accumulator.astype(np.uint64)
        ) & np.uint64(0xFFFF)
        observed = np.ascontiguousarray(observed.astype(np.uint16))
        prediction = estimate_apu_g2_u16_gemm_wall_us(
            lhs.shape[0],
            lhs.shape[1],
            rhs.shape[1],
            batch_columns=batch_columns,
        )
        base = float(batch_columns * 100 + invocation * 10)
        host_timings = {
            "h2d": base + 0.1,
            "host_task": base + 0.2,
            "d2h": base + 0.3,
            "end_to_end": 3 * base + 1.0,
        }
        source_hashes = dict(provenance["source_sha256"])
        runtime_sha256 = provenance["runtime_driver_sha256"]
        if fault == "missing_host_field" and invocation == 0:
            host_timings.pop("d2h")
        if fault == "mixed_source" and invocation == 1:
            path = next(iter(source_hashes))
            source_hashes[path] = (
                "0" * 64 if source_hashes[path] != "0" * 64 else "1" * 64
            )
        if fault == "incorrect_output" and invocation == 0:
            observed[0, 0] ^= np.uint16(1)
        return RunResult(
            cycles=10_000 + batch_columns * 100 + invocation,
            stdout="injected fake runner",
            backend="apu_v2",
            extra={
                "outputs": {"out": observed},
                "host_timings_us": host_timings,
                "hardware_tasks": prediction["hardware_tasks"],
                "weight_uploads": prediction["weight_uploads"],
                "schedule": {
                    "kind": "column_batched_u16_gemm",
                    "batch_columns": batch_columns,
                    "reduction_tile": 128,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                },
                "project": {
                    "source_sha256": source_hashes,
                    "runtime_sha256": runtime_sha256,
                    "promotion_platform_fingerprint": (
                        None
                        if provenance["platform_contract"] is None
                        else (
                            provenance["platform_contract"]["schema"],
                            provenance["platform_contract"]["target"],
                            provenance["platform_contract"]["hardware_family"],
                            provenance["platform_contract"]["fingerprint"],
                        )
                    ),
                },
            },
        )

    return runner, calls


def _self_rehash(report):
    body = {
        key: value for key, value in report.items() if key != "measurement_fingerprint"
    }
    report["measurement_fingerprint"] = MEASURER._fingerprint(
        "apu-g2-persistent-gemm-candidate-measurements-v1", body
    )


def _self_rehash_candidate_identity(candidate):
    identity = candidate["identity"]
    identity["prediction_fingerprint"] = MEASURER._fingerprint(
        "apu-g2-persistent-gemm-model-prediction-v1",
        candidate["model_prediction"],
    )
    identity["materialization_fingerprint"] = (
        MEASURER._recompute_materialization_fingerprint(candidate)
    )
    identity_body = {
        field: identity[field]
        for field in (
            "schedule_fingerprint",
            "materialization_fingerprint",
            "semantic_fingerprint",
            "source_fingerprint",
            "model_fingerprint",
            "prediction_fingerprint",
        )
    }
    identity["candidate_fingerprint"] = MEASURER._fingerprint(
        "apu-g2-persistent-gemm-measurement-candidate-v1", identity_body
    )


def test_measurement_is_deterministic_and_discards_exactly_one_warmup():
    provenance = MEASURER.current_runtime_provenance()
    first_runner, first_calls = _fake_runner(provenance)
    first = MEASURER.measure_apu_g2_persistent_candidates(
        2,
        3,
        4,
        alpha=5,
        beta=7,
        batch_columns=(3, 1),
        samples=7,
        runner=first_runner,
    )
    second_runner, second_calls = _fake_runner(provenance)
    second = MEASURER.measure_apu_g2_persistent_candidates(
        2,
        3,
        4,
        alpha=5,
        beta=7,
        batch_columns=(1, 3),
        samples=7,
        runner=second_runner,
    )

    assert first == second
    assert first["promotion_eligibility"] == MEASURER._promotion_eligibility(
        first["provenance"]["platform_contract"]
    )
    assert first["rejections"] == []
    assert first["candidate_domain"] == {
        "full_batch_columns": list(MEASURER.DEFAULT_BATCH_COLUMNS),
        "selected_batch_columns": [1, 3],
        "runtime_min": 1,
        "runtime_max": 31,
        "reduction_tile": 128,
        "full_candidate_count": 31,
        "selected_candidate_count": 2,
        "candidate_domain_complete": False,
    }
    assert Counter(call["batch_columns"] for call in first_calls) == {1: 8, 3: 8}
    assert first_calls == second_calls
    assert [
        candidate["schedule"]["batch_columns"] for candidate in first["candidates"]
    ] == [1, 3]
    assert first["measurement_protocol"] == {
        "warmup_runs_per_candidate": 1,
        "warmup_samples_retained": 0,
        "measured_samples_per_candidate": 7,
        "correctness": "bit_exact_numpy_uint16_mod_2^16_each_invocation",
        "sample_correlation": (
            "one sample object contains device and host timings from one invocation"
        ),
    }
    first_candidate = first["candidates"][0]
    assert first_candidate["samples"][0]["host_us"]["h2d_us"] == 110.1
    assert all(
        sample["sample_index"] == index
        for index, sample in enumerate(first_candidate["samples"])
    )
    assert len(first_candidate["identity"]["schedule_fingerprint"]) == 64
    assert len(first_candidate["identity"]["materialization_fingerprint"]) == 64
    assert len(first_candidate["identity"]["source_fingerprint"]) == 64
    assert len(first_candidate["identity"]["model_fingerprint"]) == 16
    identity_text = json.dumps(
        [candidate["identity"] for candidate in first["candidates"]]
    ).lower()
    assert "workload" not in identity_text
    assert "kernel" not in identity_text
    assert "tag" not in identity_text
    assert json.loads(MEASURER.report_to_json(first)) == first
    assert MEASURER.DEFAULT_BATCH_COLUMNS == tuple(range(1, 32))

    malformed = copy.deepcopy(first)
    malformed["candidates"][0]["identity"]["workload_name"] = "diagnostic"
    with pytest.raises(MEASURER.CandidateMeasurementError, match="malformed keys"):
        MEASURER.report_to_json(malformed)


def test_structural_materialization_matches_name_independent_compiler_identity():
    target = build_apu_g2_target()
    schedule = APUG2PersistentGemmSchedule(2, 128)
    materializations = []
    for workload in (structural_gemm, renamed_structural_gemm):
        module = allo.customize(workload, enable_tensor=False).module
        plan = plan_apu_g2_rank_n_contraction(module)
        materializations.append(
            _materialize_apu_g2_persistent_gemm_schedule(plan, target, schedule)
        )
    prepared = MEASURER._prepare_campaign(2, 3, 4, 1, 1, (2,)).candidates[0]

    assert (
        materializations[0].semantic_fingerprint
        == materializations[1].semantic_fingerprint
    )
    assert (
        materializations[0].promotion_materialization_fingerprint
        == materializations[1].promotion_materialization_fingerprint
    )
    assert (
        prepared.materialized.semantic_fingerprint
        == materializations[0].semantic_fingerprint
    )
    assert (
        prepared.identity["materialization_fingerprint"]
        == materializations[0].promotion_materialization_fingerprint
    )
    parameters = inspect.signature(
        MEASURER.measure_apu_g2_persistent_candidates
    ).parameters
    assert not ({"workload", "kernel", "name", "tag"} & set(parameters))


def test_duplicate_candidates_and_insufficient_samples_fail_before_execution():
    calls = []

    def runner(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("runner must not be called")

    with pytest.raises(MEASURER.CandidateMeasurementError, match="at least 7"):
        MEASURER.measure_apu_g2_persistent_candidates(
            2, 3, 4, batch_columns=(1,), samples=6, runner=runner
        )
    with pytest.raises(MEASURER.CandidateMeasurementError, match="duplicate"):
        MEASURER.measure_apu_g2_persistent_candidates(
            2, 3, 4, batch_columns=(1, 1), samples=7, runner=runner
        )
    assert calls == []


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("missing_host_field", "positive finite number"),
        ("mixed_source", "mixed persistent-runtime source provenance"),
        ("incorrect_output", "failed uint16 correctness"),
    ),
)
def test_malformed_mixed_or_incorrect_runner_results_fail_closed(fault, message):
    provenance = MEASURER.current_runtime_provenance()
    runner, _calls = _fake_runner(provenance, fault=fault)

    with pytest.raises(MEASURER.CandidateMeasurementError, match=message):
        MEASURER.measure_apu_g2_persistent_candidates(
            2, 3, 4, batch_columns=(1,), samples=7, runner=runner
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("prediction", "identity is stale"),
        ("materialization", "materialization is malformed"),
        ("domain", "candidate_domain_complete is false"),
        ("sample_count", "insufficient timing samples"),
        ("sample_correlation", "host phases are uncorrelated"),
        ("platform", "current exact campaign"),
    ),
)
def test_current_evidence_cannot_be_forged_by_self_rehash(mutation, message):
    provenance = MEASURER.current_runtime_provenance()
    runner, _calls = _fake_runner(provenance)
    report = MEASURER.measure_apu_g2_persistent_candidates(
        2, 3, 4, batch_columns=(1, 3), samples=7, runner=runner
    )
    forged = copy.deepcopy(report)
    candidate = forged["candidates"][0]
    if mutation == "prediction":
        candidate["model_prediction"]["h2d_us"] += 1.0
        candidate["model_prediction"]["wall_us"] += 1.0
        _self_rehash_candidate_identity(candidate)
    elif mutation == "materialization":
        candidate["materialization"]["beta"] += 1
        _self_rehash_candidate_identity(candidate)
    elif mutation == "domain":
        forged["candidate_domain"]["candidate_domain_complete"] = True
    elif mutation == "sample_count":
        candidate["samples"].pop()
    elif mutation == "sample_correlation":
        candidate["samples"][0]["host_us"]["wall_us"] += 1.0
    else:
        forged["provenance"]["platform_contract"] = {
            "schema": MEASURER._PLATFORM_SCHEMA,
            "target": MEASURER._PLATFORM_TARGET,
            "hardware_family": MEASURER._PLATFORM_HARDWARE_FAMILY,
            "fingerprint": "0" * 64,
        }
        forged["promotion_eligibility"] = MEASURER._promotion_eligibility(
            forged["provenance"]["platform_contract"]
        )
    _self_rehash(forged)

    with pytest.raises(MEASURER.CandidateMeasurementError, match=message):
        MEASURER.validate_measurement_report(forged)


def test_cli_refuses_overwrite_unless_explicitly_requested(tmp_path, capsys):
    output = tmp_path / "measurements.json"
    output.write_text("sentinel\n", encoding="utf-8")
    calls = []

    def forbidden_runner(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("existing output must be rejected before execution")

    assert (
        MEASURER.main(
            ["2", "3", "4", "--batch-columns", "1", "--output", str(output)],
            runner=forbidden_runner,
        )
        == 1
    )
    assert calls == []
    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert "refusing to overwrite" in capsys.readouterr().err

    runner, calls = _fake_runner(MEASURER.current_runtime_provenance())
    assert (
        MEASURER.main(
            [
                "2",
                "3",
                "4",
                "--alpha",
                "5",
                "--beta",
                "7",
                "--batch-columns",
                "1",
                "--output",
                str(output),
                "--overwrite",
            ],
            runner=runner,
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    MEASURER.validate_measurement_report(report)
    assert len(calls) == 8
