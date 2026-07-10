"""Focused hardware-free tests for APUv1 profile-candidate measurement."""

from dataclasses import replace
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from allo.pim.apu_v1_profile_search import (
    derive_apu_v1_profile_plan_decisions,
    derive_apu_v1_profile_row_tiles,
)
from allo.pim.targets import build_apu_v1_target


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "measure_apu_v1_profile_candidates.py"
SPEC = importlib.util.spec_from_file_location(
    "measure_apu_v1_profile_candidates", SCRIPT
)
MEASURE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MEASURE
SPEC.loader.exec_module(MEASURE)


def _fingerprint(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class _FakeHooks:
    def __init__(self, *, duplicate_samples=False):
        self.compile_requests = []
        self.run_count = 0
        self.plan_display_names = set()
        self.duplicate_samples = duplicate_samples

    def compiler(self, request):
        self.compile_requests.append(request)
        self.plan_display_names.add(request.realization.plan.name)
        return MEASURE.CompiledProfileShard(
            request.wave_kind,
            request.emitted_source_fingerprint,
            request.materialization_fingerprint,
            request.target_fingerprint,
            request.cost_fingerprint,
            request.runtime_source_fingerprint,
            request.platform_contract,
            request.realization,
        )

    def runner(self, compiled, inputs):
        self.run_count += 1
        outputs = MEASURE._expected_outputs(inputs)
        sample_token = "duplicate" if self.duplicate_samples else self.run_count
        base = 1000 if compiled.wave_kind == "full" else 200
        return MEASURE.BoardExecution(
            _fingerprint(f"sample-{sample_token}"),
            base + self.run_count,
            outputs,
            compiled.emitted_source_fingerprint,
            compiled.materialization_fingerprint,
            compiled.target_fingerprint,
            compiled.cost_fingerprint,
            compiled.runtime_source_fingerprint,
            compiled.platform_contract,
        )


def _small_plan_fingerprint():
    target = build_apu_v1_target()
    analysis = MEASURE._structural_analysis(32, 8, 8)
    return derive_apu_v1_profile_plan_decisions(analysis, target, row_tile=32)[
        0
    ].physical_fingerprint


def _partial_plan_fingerprint():
    target = build_apu_v1_target()
    analysis = MEASURE._structural_analysis(96, 16, 1024)
    return derive_apu_v1_profile_plan_decisions(analysis, target, row_tile=64)[
        0
    ].physical_fingerprint


def _measure_small(hooks, **kwargs):
    return MEASURE.measure_apu_v1_profile_candidates(
        32,
        8,
        8,
        row_tiles=(32,),
        physical_plan_fingerprints=(_small_plan_fingerprint(),),
        measured_samples=5,
        compiler_hook=hooks.compiler,
        runner_hook=hooks.runner,
        **kwargs,
    )


def _self_rehash(report):
    for candidate in report["candidates"]:
        body = {
            key: value
            for key, value in candidate.items()
            if key != "measurement_fingerprint"
        }
        candidate["measurement_fingerprint"] = MEASURE._fingerprint(
            "apu-v1-profile-candidate-measurement-v1", body
        )
    body = {key: value for key, value in report.items() if key != "report_fingerprint"}
    report["report_fingerprint"] = MEASURE._fingerprint(
        "apu-v1-profile-candidate-measurement-report-v1", body
    )


def test_default_domain_is_structural_strict_and_name_free():
    hooks = _FakeHooks()
    report = MEASURE.measure_apu_v1_profile_candidates(
        32,
        8,
        8,
        measured_samples=5,
        compiler_hook=hooks.compiler,
        runner_hook=hooks.runner,
    )
    analysis = MEASURE._structural_analysis(32, 8, 8)
    target = build_apu_v1_target()
    row_tiles = derive_apu_v1_profile_row_tiles(analysis, target)
    expected_candidates = sum(
        len(derive_apu_v1_profile_plan_decisions(analysis, target, row_tile=row_tile))
        for row_tile in row_tiles
    )

    assert report["schema"] == MEASURE.SCHEMA
    platform_contract = report["provenance"]["platform_contract"]
    assert report["promotion_eligibility"] == MEASURE._promotion_eligibility(
        platform_contract
    )
    assert report["logical_shape"] == {"M": 32, "K": 8, "N": 8}
    assert all(
        candidate["provenance"]["platform_contract"] == platform_contract
        and candidate["waves"]["full"]["platform_contract"] == platform_contract
        for candidate in report["candidates"]
    )
    assert report["domain"] == {
        "derived_row_tiles": [32],
        "selected_row_tiles": [32],
        "selected_physical_plan_fingerprints": None,
        "row_tile_domain_complete": True,
        "physical_plan_domain_complete": True,
        "full_domain_candidate_count": expected_candidates,
        "selected_candidate_count": expected_candidates,
        "materialized_candidate_count": expected_candidates,
        "rejected_candidate_count": 0,
        "measured_candidate_count": expected_candidates,
        "candidate_domain_complete": True,
    }
    assert len(report["candidates"]) == expected_candidates
    assert hooks.run_count == expected_candidates * 6
    assert len(hooks.compile_requests) == expected_candidates
    assert all(
        len(candidate["correlated_cycle_samples"]) == 5
        and candidate["warmup"]["count"] == 1
        and candidate["warmup"]["discarded"] is True
        and candidate["waves"]["partial"] is None
        for candidate in report["candidates"]
    )
    rendered = MEASURE.report_to_json(report)
    parsed = json.loads(rendered)
    assert parsed == report
    assert "structural_profile_measurement" not in rendered
    assert "operand_0" not in rendered
    assert "output_0" not in rendered
    assert all(f'"{name}"' not in rendered for name in hooks.plan_display_names)
    for candidate in report["candidates"]:
        identity_json = json.dumps(candidate["candidate_identity"], sort_keys=True)
        assert "source_fingerprint" not in identity_json
        assert "materialization_fingerprint" not in identity_json
        assert set(candidate) == MEASURE._CANDIDATE_FIELDS


def test_partial_wave_samples_are_correlated_and_composed_exactly():
    hooks = _FakeHooks()
    report = MEASURE.measure_apu_v1_profile_candidates(
        96,
        16,
        1024,
        row_tiles=(64,),
        physical_plan_fingerprints=(_partial_plan_fingerprint(),),
        measured_samples=5,
        compiler_hook=hooks.compiler,
        runner_hook=hooks.runner,
    )
    candidate = report["candidates"][0]

    assert [request.wave_kind for request in hooks.compile_requests] == [
        "full",
        "partial",
    ]
    assert hooks.run_count == 12
    assert candidate["waves"]["full"]["kind"] == "full"
    assert candidate["waves"]["full"]["shape"] == {"M": 64, "K": 16, "N": 1024}
    assert candidate["waves"]["partial"]["kind"] == "partial"
    assert candidate["waves"]["partial"]["shape"] == {
        "M": 32,
        "K": 16,
        "N": 1024,
    }
    assert candidate["composition"] == {
        "row_waves": 2,
        "full_wave_count": 1,
        "partial_wave_count": 1,
        "column_repetitions": 1,
        "full_wave_executions": 1,
        "partial_wave_executions": 1,
        "total_wave_executions": 2,
    }
    for index, sample in enumerate(candidate["correlated_cycle_samples"]):
        assert sample["index"] == index
        assert sample["composed_cycles"] == (
            sample["full_wave"]["cycles"] + sample["partial_wave"]["cycles"]
        )
    assert candidate["correctness"] == {
        "status": "pass",
        "oracle": "bit-exact structural uint16 contraction",
        "checked_board_executions": 12,
    }
    fingerprints = [
        candidate["provenance"]["profile_materialization_fingerprint"],
        candidate["provenance"]["target_fingerprint"],
        candidate["provenance"]["cost_fingerprint"],
    ]
    for wave in candidate["waves"].values():
        fingerprints.extend(
            wave[field]
            for field in (
                "emitted_source_fingerprint",
                "structural_source_fingerprint",
                "plan_emitted_source_fingerprint",
                "materialization_fingerprint",
            )
        )
    assert all(len(value) in {16, 64} for value in fingerprints)


def test_subsets_samples_and_duplicate_evidence_fail_closed():
    hooks = _FakeHooks()
    with pytest.raises(MEASURE.ProfileMeasurementError, match="at least 5"):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            measured_samples=4,
            compiler_hook=hooks.compiler,
            runner_hook=hooks.runner,
        )
    with pytest.raises(MEASURE.ProfileMeasurementError, match="row-tile.*duplicates"):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            row_tiles=(32, 32),
            measured_samples=5,
            compiler_hook=hooks.compiler,
            runner_hook=hooks.runner,
        )
    plan = _small_plan_fingerprint()
    with pytest.raises(
        MEASURE.ProfileMeasurementError, match="physical-plan.*duplicates"
    ):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            physical_plan_fingerprints=(plan, plan),
            measured_samples=5,
            compiler_hook=hooks.compiler,
            runner_hook=hooks.runner,
        )
    with pytest.raises(MEASURE.ProfileMeasurementError, match="outside.*domain"):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            physical_plan_fingerprints=("f" * 64,),
            measured_samples=5,
            compiler_hook=hooks.compiler,
            runner_hook=hooks.runner,
        )
    duplicate_hooks = _FakeHooks(duplicate_samples=True)
    with pytest.raises(MEASURE.ProfileMeasurementError, match="duplicate board sample"):
        _measure_small(duplicate_hooks)

    subset = _measure_small(_FakeHooks())
    assert subset["domain"]["candidate_domain_complete"] is False
    assert subset["domain"]["row_tile_domain_complete"] is True
    assert subset["domain"]["physical_plan_domain_complete"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("prediction", "model prediction is stale"),
        ("source", "wave evidence is stale"),
        ("composition", "composition is stale"),
        ("sample", "false composed cycles"),
        ("correctness", "correctness evidence is false"),
        ("platform", "provenance is stale"),
    ),
)
def test_structural_evidence_cannot_be_forged_by_self_rehash(mutation, message):
    report = copy.deepcopy(_measure_small(_FakeHooks()))
    candidate = report["candidates"][0]
    if mutation == "prediction":
        candidate["model_prediction"]["composed_cycles"] += 1
    elif mutation == "source":
        candidate["waves"]["full"]["runtime_source_fingerprint"] = "0" * 64
    elif mutation == "composition":
        candidate["composition"]["row_waves"] += 1
    elif mutation == "sample":
        candidate["correlated_cycle_samples"][0]["composed_cycles"] += 1
    elif mutation == "correctness":
        candidate["correctness"]["checked_board_executions"] += 1
    else:
        contract = {
            "schema": MEASURE._PLATFORM_SCHEMA,
            "target": MEASURE._PLATFORM_TARGET,
            "hardware_family": MEASURE._PLATFORM_HARDWARE_FAMILY,
            "fingerprint": "0" * 64,
        }
        candidate["provenance"]["platform_contract"] = contract
        report["provenance"]["platform_contract"] = contract
        report["promotion_eligibility"] = MEASURE._promotion_eligibility(contract)
    _self_rehash(report)

    with pytest.raises(MEASURE.ProfileMeasurementError, match=message):
        MEASURE.validate_measurement_report(report)


def test_compiler_and_runner_provenance_fail_closed():
    hooks = _FakeHooks()

    def mixed_compiler(request):
        return replace(hooks.compiler(request), target_fingerprint="0" * 64)

    with pytest.raises(
        MEASURE.ProfileMeasurementError, match="compiler.*mixed provenance"
    ):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            row_tiles=(32,),
            physical_plan_fingerprints=(_small_plan_fingerprint(),),
            measured_samples=5,
            compiler_hook=mixed_compiler,
            runner_hook=hooks.runner,
        )

    hooks = _FakeHooks()

    def mixed_runner(compiled, inputs):
        return replace(hooks.runner(compiled, inputs), cost_fingerprint="0" * 64)

    with pytest.raises(
        MEASURE.ProfileMeasurementError, match="execution.*mixed provenance"
    ):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            row_tiles=(32,),
            physical_plan_fingerprints=(_small_plan_fingerprint(),),
            measured_samples=5,
            compiler_hook=hooks.compiler,
            runner_hook=mixed_runner,
        )


def test_every_execution_must_pass_bit_exact_correctness():
    hooks = _FakeHooks()

    def incorrect_runner(compiled, inputs):
        execution = hooks.runner(compiled, inputs)
        wrong = {"output_0": np.asarray(execution.outputs["output_0"]).copy()}
        wrong["output_0"].flat[0] ^= np.uint16(1)
        return replace(execution, outputs=wrong)

    with pytest.raises(MEASURE.ProfileMeasurementError, match="bit-exact correctness"):
        MEASURE.measure_apu_v1_profile_candidates(
            32,
            8,
            8,
            row_tiles=(32,),
            physical_plan_fingerprints=(_small_plan_fingerprint(),),
            measured_samples=5,
            compiler_hook=hooks.compiler,
            runner_hook=incorrect_runner,
        )


def test_strict_json_overwrite_and_cli_argument_forwarding(monkeypatch, tmp_path):
    hooks = _FakeHooks()
    report = _measure_small(hooks)
    output = tmp_path / "measurements.json"
    MEASURE.write_measurement_report(output, report)
    original = output.read_text(encoding="utf-8")
    with pytest.raises(MEASURE.ProfileMeasurementError, match="--overwrite"):
        MEASURE.write_measurement_report(output, report)
    MEASURE.write_measurement_report(output, report, overwrite=True)
    assert output.read_text(encoding="utf-8") == original

    malformed = {**report, "kernel": "forbidden-display-name"}
    with pytest.raises(MEASURE.ProfileMeasurementError, match="unsupported kernel"):
        MEASURE.report_to_json(malformed)

    captured = {}

    def fake_measure(*args, **kwargs):
        captured["measure"] = (args, kwargs)
        return report

    def fake_write(path, value, *, overwrite=False):
        captured["write"] = (path, value, overwrite)

    monkeypatch.setattr(MEASURE, "measure_apu_v1_profile_candidates", fake_measure)
    monkeypatch.setattr(MEASURE, "write_measurement_report", fake_write)
    cli_output = tmp_path / "cli.json"
    status = MEASURE.main(
        [
            "96",
            "16",
            "1024",
            "--row-tile",
            "64",
            "--physical-plan",
            "a" * 64,
            "--samples",
            "5",
            "--output",
            str(cli_output),
            "--overwrite",
        ]
    )

    assert status == 0
    assert captured["measure"] == (
        (96, 16, 1024),
        {
            "row_tiles": [64],
            "physical_plan_fingerprints": ["a" * 64],
            "measured_samples": 5,
        },
    )
    assert captured["write"] == (cli_output, report, True)
