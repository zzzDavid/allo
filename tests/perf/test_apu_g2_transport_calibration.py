"""Focused tests for the APUg2 transport-calibration artifact report."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "report_apu_g2_transport_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "report_apu_g2_transport_calibration", SCRIPT
)
REPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPORTER
SPEC.loader.exec_module(REPORTER)


def _write_json(path, value, *, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent) + "\n", encoding="utf-8")
    return path


def _profile(
    leaf,
    token,
    shape,
    *,
    scale=1.0,
    schedule=None,
    alpha=1,
    beta=1,
):
    schedule = schedule or {
        "kind": REPORTER.SCHEDULE_KIND,
        "batch_columns": 31,
        "reduction_tile": 128,
        "resident_accumulator": True,
        "contiguous_readback": True,
    }
    prediction = REPORTER.estimate_apu_g2_u16_gemm_wall_us(
        *shape,
        batch_columns=schedule["batch_columns"],
        reduction_tile=schedule["reduction_tile"],
    )
    phases = {
        name: round(float(prediction[f"{name}_us"]) * scale, 6)
        for name in ("h2d", "host_task", "d2h")
    }
    measured_wall = math.fsum(phases.values())
    board_relative = f"profile/{token}.board.log"
    record_relative = f"profile/{token}.json"
    board_path = leaf / board_relative
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text("measured board profile\n", encoding="utf-8")
    record_path = _write_json(
        leaf / record_relative,
        {
            "backend": "apu_v2",
            "cycles": 123,
            "extra": {
                "hardware_tasks": prediction["hardware_tasks"],
                "weight_uploads": prediction["weight_uploads"],
                "host_timings_us": phases,
                "schedule": schedule,
            },
            "outputs": {
                "out": {
                    "dtype": "uint16",
                    "shape": [shape[0], shape[2]],
                }
            },
        },
    )
    return (
        {
            "alpha": alpha,
            "batch_columns": schedule["batch_columns"],
            "beta": beta,
            "correctness": "PASS complete stage bit-exact modulo 2^16",
            "count": 1,
            "kind": REPORTER.PROFILE_KIND,
            "profile": board_relative,
            "profile_hardware_tasks": prediction["hardware_tasks"],
            "profile_shape": {"M": shape[0], "K": shape[1], "N": shape[2]},
            "profile_wall_microseconds": measured_wall,
        },
        record_path,
    )


def _result(root, directory, profiles, *, kernel=None, backend="apu-v2"):
    leaf = root / directory
    path = _write_json(
        leaf / "result.json",
        {
            "backend": backend,
            "dataset": "structural fixture",
            "dtype": "uint16",
            "framework": "tenon",
            "kernel": kernel or directory,
            "result": {
                "profiles": profiles,
                "wall_metric": REPORTER.WALL_METRIC,
            },
            "status": "MEASURED-COMPOSITION",
        },
    )
    return path


def _artifact_tree(root):
    paths = {"relevant_results": [], "profile_records": {}}

    dense_leaf = root / "z-dense-anchor"
    dense, dense_record = _profile(dense_leaf, "dense", REPORTER.TRAINING_ANCHORS[0])
    paths["relevant_results"].append(
        _result(root, "z-dense-anchor", [dense], kernel="arbitrary-dense-name")
    )
    paths["profile_records"]["dense"] = dense_record

    for directory in ("b-vector-copy", "a-vector-copy"):
        leaf = root / directory
        vector, vector_record = _profile(
            leaf,
            "vector",
            REPORTER.TRAINING_ANCHORS[1],
        )
        paths["relevant_results"].append(
            _result(root, directory, [vector], kernel=f"renamable-{directory}")
        )
        paths["profile_records"][directory] = vector_record

    first_heldout = (64, 96, 3)
    for directory, scale in (
        ("heldout-copy-one", 0.8),
        ("heldout-copy-two", 0.8),
        ("heldout-rerun", 1.2),
    ):
        leaf = root / directory
        profile, profile_record = _profile(leaf, "matrix", first_heldout, scale=scale)
        paths["relevant_results"].append(
            _result(root, directory, [profile], kernel=f"source-{directory}")
        )
        paths["profile_records"][directory] = profile_record

    second_heldout = (128, 160, 33)
    leaf = root / "heldout-second-shape"
    profile, profile_record = _profile(leaf, "matrix", second_heldout, scale=1.1)
    profile["count"] = 4
    paths["relevant_results"].append(
        _result(root, "heldout-second-shape", [profile], kernel="another-name")
    )
    paths["profile_records"]["second-heldout"] = profile_record

    _write_json(
        root / "mixed" / "other-backend" / "result.json",
        {"backend": "apu-v1", "result": "not an APUv2 record"},
    )
    _write_json(
        root / "mixed" / "other-profile" / "result.json",
        {
            "backend": "apu-v2",
            "result": {"profiles": [{"kind": "dot_tile"}]},
        },
    )
    paths["first_heldout"] = first_heldout
    paths["second_heldout"] = second_heldout
    return paths


def _all_records(report):
    return report["training_records"] + report["heldout_records"]


def _record_for_shape(report, shape):
    return next(
        record
        for record in _all_records(report)
        if tuple(record["shape"][key] for key in ("M", "K", "N")) == shape
    )


def _records_for_shape(report, shape):
    return [
        record
        for record in _all_records(report)
        if tuple(record["shape"][key] for key in ("M", "K", "N")) == shape
    ]


def test_structural_deduplication_pointwise_report_and_fingerprints(tmp_path):
    paths = _artifact_tree(tmp_path)

    first = REPORTER.build_transport_calibration_report(tmp_path)
    second = REPORTER.build_transport_calibration_report(tmp_path)

    assert first == second
    assert REPORTER.report_to_json(first) == REPORTER.report_to_json(second)
    assert first["artifact_summary"] == {
        "source_result_record_count": 7,
        "profile_observation_count": 7,
        "unique_structural_record_count": 4,
        "training_record_count": 2,
        "heldout_record_count": 2,
    }
    assert first["model"]["fit_verification"]["status"] == ("matches_current_model")
    assert first["methodology"]["declared_anchor_fit_reproduced"] is True
    assert [record["shape"] for record in first["training_records"]] == [
        {"M": 1000, "K": 1200, "N": 1100},
        {"M": 1900, "K": 2100, "N": 1},
    ]
    assert [record["shape"] for record in first["heldout_records"]] == [
        {"M": 64, "K": 96, "N": 3},
        {"M": 128, "K": 160, "N": 33},
    ]

    vector = _record_for_shape(first, REPORTER.TRAINING_ANCHORS[1])
    assert vector["observation_count"] == 2
    assert vector["unique_measurement_count"] == 1
    assert len(vector["provenance"]) == 2

    heldout = _record_for_shape(first, paths["first_heldout"])
    assert heldout["observation_count"] == 3
    assert heldout["unique_measurement_count"] == 2
    assert heldout["measured"]["wall_us"] == pytest.approx(
        heldout["predicted"]["wall_us"] * 0.8, abs=1e-5
    )
    assert heldout["measurement_aggregation"] == (
        "median_wall_correlated_observation"
    )

    cross_shape = first["cross_shape_prediction"]
    assert cross_shape["status"] == "available"
    assert cross_shape["heldout_error"]["record_count"] == 2
    assert cross_shape["coverage"]["prediction_coverage"] == 1.0
    assert cross_shape["coverage"]["cross_shape_extrapolation_coverage"] == 1.0
    ranking = first["within_shape_candidate_ranking"]
    assert ranking["status"] == "unavailable"
    assert ranking["metrics"] is None
    assert ranking["observed_candidate_count_per_semantic_problem"] == 1
    assert ranking["observed_candidate_count_per_shape"] == 1

    fingerprints = [
        first["report_fingerprint"],
        first["model"]["model_fingerprint"],
        first["partition"]["partition_fingerprint"],
        *(record["structural_identity_fingerprint"] for record in _all_records(first)),
        *(record["evidence_fingerprint"] for record in _all_records(first)),
    ]
    assert all(len(value) == 64 for value in fingerprints)
    assert all(set(value) <= set("0123456789abcdef") for value in fingerprints)

    dense_path = paths["relevant_results"][0]
    dense_data = json.loads(dense_path.read_text(encoding="utf-8"))
    dense_path.write_text(
        json.dumps(dense_data, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert REPORTER.build_transport_calibration_report(tmp_path) == first


def test_benchmark_names_are_provenance_not_model_identity(tmp_path):
    paths = _artifact_tree(tmp_path)
    before = REPORTER.build_transport_calibration_report(tmp_path)

    for index, result_path in enumerate(paths["relevant_results"]):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["kernel"] = f"completely-renamed-{index}"
        _write_json(result_path, data)

    after = REPORTER.build_transport_calibration_report(tmp_path)

    assert after["model"] == before["model"]
    assert (
        after["partition"]["partition_fingerprint"]
        == before["partition"]["partition_fingerprint"]
    )
    assert [
        record["structural_identity_fingerprint"] for record in _all_records(after)
    ] == [record["structural_identity_fingerprint"] for record in _all_records(before)]
    assert after["report_fingerprint"] != before["report_fingerprint"]
    assert "completely-renamed" not in json.dumps(after["model"])


def test_coefficient_distinct_programs_are_not_collapsed_by_shape(tmp_path):
    paths = _artifact_tree(tmp_path)
    shape = paths["first_heldout"]
    leaf = tmp_path / "heldout-different-epilogue"
    profile, _record = _profile(
        leaf,
        "matrix-scaled",
        shape,
        scale=0.9,
        alpha=5,
        beta=4,
    )
    _result(root=tmp_path, directory="heldout-different-epilogue", profiles=[profile])

    report = REPORTER.build_transport_calibration_report(tmp_path)
    records = _records_for_shape(report, shape)

    assert len(records) == 2
    assert {tuple(record["epilogue"].values()) for record in records} == {
        (1, 1),
        (5, 4),
    }
    assert len({record["semantic_problem_fingerprint"] for record in records}) == 2
    assert len({record["structural_identity_fingerprint"] for record in records}) == 2
    assert len({record["schedule_fingerprint"] for record in records}) == 1
    assert report["artifact_summary"]["unique_structural_record_count"] == 5
    assert report["artifact_summary"]["heldout_record_count"] == 3
    assert report["within_shape_candidate_ranking"]["shape_count"] == 4
    assert report["within_shape_candidate_ranking"]["semantic_problem_count"] == 5


def test_measurement_aggregation_preserves_correlated_phase_sum():
    measurements = (
        REPORTER._Measurement(1.0, 1.0, 1.0, 3.0),
        REPORTER._Measurement(100.0, 1.0, 1.0, 102.0),
        REPORTER._Measurement(1.0, 100.0, 100.0, 201.0),
    )

    selected = REPORTER._median_measurement(measurements)

    assert selected is measurements[1]
    assert math.fsum(
        (selected.h2d_us, selected.host_task_us, selected.d2h_us)
    ) == selected.wall_us


def test_malformed_and_wrong_backend_records_fail_closed(tmp_path):
    _artifact_tree(tmp_path)
    malformed = tmp_path / "mixed" / "malformed" / "result.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(
        REPORTER.ArtifactReportError, match="cannot load artifact result"
    ):
        REPORTER.build_transport_calibration_report(tmp_path)

    malformed.unlink()
    dense_result = tmp_path / "z-dense-anchor" / "result.json"
    data = json.loads(dense_result.read_text(encoding="utf-8"))
    data["backend"] = "apu-v1"
    _write_json(dense_result, data)
    with pytest.raises(REPORTER.ArtifactReportError, match="without APUv2 backend"):
        REPORTER.build_transport_calibration_report(tmp_path)


def test_malformed_measurement_and_mixed_schedule_fail_closed(tmp_path):
    paths = _artifact_tree(tmp_path)
    profile_path = paths["profile_records"]["second-heldout"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    del profile["extra"]["host_timings_us"]["host_task"]
    _write_json(profile_path, profile)
    with pytest.raises(REPORTER.ArtifactReportError, match="host_task"):
        REPORTER.build_transport_calibration_report(tmp_path)

    _artifact_tree(tmp_path)
    profile_path = paths["profile_records"]["second-heldout"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    del profile["extra"]["schedule"]["contiguous_readback"]
    _write_json(profile_path, profile)
    with pytest.raises(
        REPORTER.ArtifactReportError,
        match="one common structural schedule",
    ):
        REPORTER.build_transport_calibration_report(tmp_path)


def test_cli_emits_deterministic_json_and_fails_without_evidence(tmp_path, capsys):
    _artifact_tree(tmp_path)

    assert REPORTER.main([str(tmp_path)]) == 0
    first = capsys.readouterr()
    parsed = json.loads(first.out)
    assert first.err == ""
    assert first.out.endswith("\n")
    assert parsed["report_fingerprint"]

    output = tmp_path / "report.json"
    assert REPORTER.main([str(tmp_path), "--output", str(output)]) == 0
    second = capsys.readouterr()
    assert second.out == ""
    assert second.err == ""
    assert output.read_text(encoding="utf-8") == first.out

    empty = tmp_path / "empty"
    empty.mkdir()
    assert REPORTER.main([str(empty)]) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert failure.err.startswith("FAIL APUg2 transport calibration:")
    assert "no result.json" in failure.err


def test_declared_model_constants_must_reproduce_anchor_phase_fit(
    tmp_path,
    monkeypatch,
):
    _artifact_tree(tmp_path)
    monkeypatch.setitem(
        REPORTER.APUG2_GEMM_WALL_FINGERPRINT_DATA,
        "task_call_us",
        REPORTER.APUG2_GEMM_WALL_FINGERPRINT_DATA["task_call_us"] + 1.0,
    )

    with pytest.raises(
        REPORTER.ArtifactReportError,
        match="do not reproduce the declared anchor phase fit",
    ):
        REPORTER.build_transport_calibration_report(tmp_path)
