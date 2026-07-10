"""Focused tests for the APUv1 profile-calibration artifact report."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "report_apu_v1_profile_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "report_apu_v1_profile_calibration", SCRIPT
)
REPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPORTER
SPEC.loader.exec_module(REPORTER)

REAL_ARTIFACTS_ROOT = REPORTER.DEFAULT_ARTIFACTS_ROOT
REAL_CORPUS = REAL_ARTIFACTS_ROOT / REPORTER.CORPUS_RELATIVE_PATH


def _write_json(path, value, *, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent) + "\n", encoding="utf-8")
    return path


def _shape_manifest(shape):
    rows, reduction, columns = shape
    return {"M": rows, "N": columns, "K": reduction}


def _mlir(shape, function):
    rows, reduction, columns = shape
    return f"""module {{
  func.func @{function}(%left: memref<{rows}x{reduction}xi16>, %right: memref<{reduction}x{columns}xi16>, %out: memref<{rows}x{columns}xi16>) {{
    return
  }}
}}
"""


def _board_log(name, shape, plan, cycles):
    rows, reduction, columns = shape
    return f"""{REPORTER.BOARD_HEADER}
hardware: {REPORTER.BOARD_HARDWARE}
kernel: {name}
physical_shape: M={rows} N={columns} K={reduction}
selected_plan: {plan}
correctness: PASS bit-exact modulo 2^16
cycles: {cycles}
microseconds_at_500MHz: {cycles / 500.0}
counter_source: {REPORTER.BOARD_COUNTER_SOURCE}
TIME STAMP: synthetic UTC
ARCK[0]: Running synthetic task
ARCT[0]: total - hits:1 crun:{cycles}
"""


def _candidate(
    semantic_shape,
    profile_columns,
    tile_m,
    plan,
    shard_cycles,
    *,
    tile_count=1,
    operation_count=1,
):
    rows, _reduction, _columns = semantic_shape
    row_waves = math.ceil(rows / tile_m)
    repeats = row_waves * tile_count * operation_count
    return {
        "tile_m": tile_m,
        "row_waves": row_waves,
        "plan": plan,
        "estimated_shard_cycles": shard_cycles,
        "estimated_composed_cycles": shard_cycles * repeats,
    }


def _measured_result(
    artifacts_root,
    source,
    semantic_shape,
    profile_shape,
    measured_cycles,
    *,
    plan,
    predicted_cycles=None,
    challenger_tile=None,
    challenger_cycles=None,
    operation_count=1,
    tile_count=1,
):
    corpus = artifacts_root / REPORTER.CORPUS_RELATIVE_PATH
    leaf = corpus / source
    rows, _reduction, columns = semantic_shape
    profile_rows, _profile_reduction, profile_columns = profile_shape
    row_waves = math.ceil(rows / profile_rows)
    composition_repeats = row_waves * tile_count * operation_count
    profile_total_cycles = measured_cycles * composition_repeats

    board_relative = f"profile/{source}.board.log"
    profile_mlir_relative = f"compiled/{source}.profile.mlir"
    gvml_relative = f"compiled/{source}.gvml.c"
    full_mlir_relative = f"compiled/{source}.full.mlir"
    (leaf / board_relative).parent.mkdir(parents=True, exist_ok=True)
    (leaf / board_relative).write_text(
        _board_log(f"declared/{source}", profile_shape, plan, measured_cycles),
        encoding="utf-8",
    )
    (leaf / profile_mlir_relative).parent.mkdir(parents=True, exist_ok=True)
    (leaf / profile_mlir_relative).write_text(
        _mlir(profile_shape, f"profile_{source}"), encoding="utf-8"
    )
    (leaf / gvml_relative).write_text(
        f"static void {plan}_vector(void) {{ gvml_init_once(); }}\n",
        encoding="utf-8",
    )
    (leaf / full_mlir_relative).write_text(
        _mlir(semantic_shape, f"full_{source}"), encoding="utf-8"
    )

    candidates = []
    if predicted_cycles is not None:
        candidates.append(
            _candidate(
                semantic_shape,
                profile_columns,
                profile_rows,
                plan,
                predicted_cycles,
                tile_count=tile_count,
                operation_count=operation_count,
            )
        )
        candidates.append(
            _candidate(
                semantic_shape,
                profile_columns,
                challenger_tile,
                f"{plan}_challenger",
                challenger_cycles,
                tile_count=tile_count,
                operation_count=operation_count,
            )
        )

    profile = {
        "name": f"profile-name-{source}",
        "profile_shape": _shape_manifest(profile_shape),
        "tile_count": tile_count,
        "row_waves": row_waves,
        "composition_repeats": composition_repeats,
        "profile_cycles": measured_cycles,
        "profile_microseconds": measured_cycles / 500.0,
        "cycles": profile_total_cycles,
        "microseconds": profile_total_cycles / 500.0,
        "plan": plan,
        "row_tile_selection": {
            "criterion": REPORTER.ROW_TILE_CRITERION,
            "candidates": candidates,
        },
        "correctness": "PASS profiled shard bit-exact modulo 2^16",
        "profile": board_relative,
        "compiled_profile_mlir": profile_mlir_relative,
        "compiled_gvml": gvml_relative,
    }
    leg = {
        "name": f"leg-name-{source}",
        "shape": _shape_manifest(semantic_shape),
        "row_waves": row_waves,
        "column_tiles": tile_count,
        "parallel_apucs": 1,
        "composition_repeats": composition_repeats,
        "operation_count": operation_count,
        "cycles": profile_total_cycles,
        "microseconds": profile_total_cycles / 500.0,
        "plan": plan,
        "correctness": "PASS every profiled shard bit-exact modulo 2^16",
        "profiles": [profile],
        "full_workload_mlir": full_mlir_relative,
    }
    revision = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return _write_json(
        leaf / "result.json",
        {
            "kernel": f"result-name-{source}",
            "backend": REPORTER.BACKEND,
            "framework": REPORTER.FRAMEWORK,
            "dataset": REPORTER.DATASET,
            "dtype": REPORTER.DTYPE,
            "status": "PASS",
            "detail": "synthetic measured contraction",
            "result": {
                "metric": REPORTER.RESULT_METRIC,
                "cycles": profile_total_cycles,
                "microseconds": profile_total_cycles / 500.0,
                "composition": REPORTER.COMPOSITION,
                "execution_scope": REPORTER.EXECUTION_SCOPE,
                "legs": [leg],
            },
            "tenon_revision": revision,
            "hardware": REPORTER.HARDWARE,
        },
    )


def _not_applicable_result(artifacts_root, source="not-applicable"):
    corpus = artifacts_root / REPORTER.CORPUS_RELATIVE_PATH
    return _write_json(
        corpus / source / "result.json",
        {
            "kernel": f"result-name-{source}",
            "backend": REPORTER.BACKEND,
            "framework": REPORTER.FRAMEWORK,
            "dataset": REPORTER.DATASET,
            "dtype": REPORTER.DTYPE,
            "status": "NOT-APPLICABLE",
            "detail": "synthetic unsupported contraction",
            "result": None,
            "tenon_revision": hashlib.sha1(source.encode("utf-8")).hexdigest(),
            "hardware": REPORTER.HARDWARE,
        },
    )


def _artifact_tree(root):
    paths = {}
    first_shape = (100, 8, 16)
    first_profile_shape = (50, 8, 16)
    paths["dense_first"] = _measured_result(
        root,
        "z-renamable-copy",
        first_shape,
        first_profile_shape,
        100,
        plan="dense_selected",
        predicted_cycles=120,
        challenger_tile=25,
        challenger_cycles=70,
    )
    paths["dense_duplicate"] = _measured_result(
        root,
        "a-second-copy",
        first_shape,
        first_profile_shape,
        120,
        plan="dense_selected",
        predicted_cycles=120,
        challenger_tile=25,
        challenger_cycles=70,
    )
    paths["dense_second"] = _measured_result(
        root,
        "second-shape",
        (80, 4, 12),
        (40, 4, 12),
        100,
        plan="other_selected",
        predicted_cycles=80,
        challenger_tile=20,
        challenger_cycles=45,
    )
    paths["singleton"] = _measured_result(
        root,
        "structural-singleton",
        (12, 7, 1),
        (12, 7, 1),
        70,
        plan="gemv_bypass",
    )
    paths["reduction_one"] = _measured_result(
        root,
        "structural-reduction-one",
        (9, 1, 6),
        (9, 1, 6),
        55,
        plan="rank_one_bypass",
        operation_count=2,
    )
    paths["not_applicable"] = _not_applicable_result(root)
    return paths


def _record_for_shape(report, category, shape):
    return next(
        record
        for record in report["profile_records"][category]
        if tuple(record["semantic_shape"][key] for key in ("M", "K", "N")) == shape
    )


def test_deterministic_structural_deduplication_metrics_and_hashes(tmp_path):
    paths = _artifact_tree(tmp_path)

    first = REPORTER.build_apu_v1_profile_calibration_report(tmp_path)
    second = REPORTER.build_apu_v1_profile_calibration_report(tmp_path)

    assert first == second
    assert REPORTER.report_to_json(first) == REPORTER.report_to_json(second)
    assert first["artifact_summary"] == {
        "source_result_record_count": 6,
        "measured_result_record_count": 5,
        "not_applicable_result_record_count": 1,
        "result_status_counts": {
            "NOT-APPLICABLE": 1,
            "PARTIAL": 0,
            "PASS": 5,
        },
        "profile_observation_count": 5,
        "unique_profile_schedule_count": 4,
        "profile_observation_counts_by_category": {
            REPORTER.DENSE_SEARCHED: 3,
            REPORTER.SINGLETON_GEMV_BYPASS: 1,
            REPORTER.REDUCTION_ONE_BYPASS: 1,
        },
        "unique_profile_schedule_counts_by_category": {
            REPORTER.DENSE_SEARCHED: 2,
            REPORTER.SINGLETON_GEMV_BYPASS: 1,
            REPORTER.REDUCTION_ONE_BYPASS: 1,
        },
        "unique_semantic_shape_counts_by_category": {
            REPORTER.DENSE_SEARCHED: 2,
            REPORTER.SINGLETON_GEMV_BYPASS: 1,
            REPORTER.REDUCTION_ONE_BYPASS: 1,
        },
        "unique_measurement_count": 5,
    }

    duplicate = _record_for_shape(first, REPORTER.DENSE_SEARCHED, (100, 8, 16))
    assert duplicate["observation_count"] == 2
    assert duplicate["unique_measurement_count"] == 2
    assert duplicate["measured"]["profile_cycles"] == 110.0
    assert duplicate["predicted"]["profile_cycles"] == 120
    assert duplicate["opaque_historical_schedule"]["tile_m"] == 50
    assert duplicate["opaque_historical_schedule"]["repetitions"] == {
        "row_waves": 2,
        "column_tiles": 1,
        "operation_count": 1,
        "composition_repeats": 2,
    }

    metrics = first["selected_candidate_cross_shape_prediction"]["error"]
    assert metrics["record_count"] == 2
    assert metrics["signed_relative_bias"] == pytest.approx(-0.05454545454545455)
    assert metrics["median_absolute_percentage_error"] == pytest.approx(
        14.545454545454547
    )
    assert metrics["p90_absolute_percentage_error"] == pytest.approx(18.90909090909091)
    assert metrics["maximum_absolute_percentage_error"] == pytest.approx(20.0)

    coverage = first["candidate_domain_coverage"]
    assert coverage["candidate_domain_count"] == 2
    assert coverage["analytical_candidate_count"] == 4
    assert coverage["challenger_candidate_count"] == 2
    assert coverage["candidate_measurement_coverage"] == 0.5
    assert coverage["challenger_measurement_coverage"] == 0.0
    ranking = first["within_shape_candidate_ranking"]
    assert ranking["status"] == "unavailable"
    assert "no device measurements" in ranking["reason"]
    assert ranking["metrics"] is None
    assert len(duplicate["selected_materializations"]) == 2
    assert all(
        materialization["promotion_eligible"] is False
        for materialization in duplicate["selected_materializations"]
    )
    assert all(
        candidate["identity_kind"] == "opaque_historical_plan_label"
        and candidate["physical_manifest_available"] is False
        and candidate["promotion_eligible"] is False
        for candidate in duplicate["candidate_domain"]["candidates"]
    )

    singleton = first["profile_records"][REPORTER.SINGLETON_GEMV_BYPASS][0]
    reduction_one = first["profile_records"][REPORTER.REDUCTION_ONE_BYPASS][0]
    assert singleton["predicted"] is singleton["pointwise_error"] is None
    assert reduction_one["predicted"] is reduction_one["pointwise_error"] is None

    fingerprints = [
        first["report_fingerprint"],
        first["content_addressing"]["historical_record_domain_fingerprint"],
        first["content_addressing"]["historical_analytical_domain_fingerprint"],
        *(
            record["historical_record_identity_fingerprint"]
            for record in first["profile_records"][REPORTER.DENSE_SEARCHED]
        ),
    ]
    assert all(len(value) == 64 for value in fingerprints)
    provenance = duplicate["provenance"][0]
    corpus = tmp_path / REPORTER.CORPUS_RELATIVE_PATH
    for label in ("board_log", "profile_mlir", "gvml"):
        source = provenance["source_files"][label]
        expected = hashlib.sha256((corpus / source["path"]).read_bytes()).hexdigest()
        assert source["sha256"] == expected

    data = json.loads(paths["dense_first"].read_text(encoding="utf-8"))
    paths["dense_first"].write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert REPORTER.build_apu_v1_profile_calibration_report(tmp_path) == first

    output = tmp_path / "report.json"
    assert REPORTER.main([str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_names_and_paths_are_provenance_not_identity(tmp_path):
    paths = _artifact_tree(tmp_path)
    before = REPORTER.build_apu_v1_profile_calibration_report(tmp_path)

    for result_index, result_path in enumerate(paths.values()):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["kernel"] = f"renamed-result-{result_index}"
        result = data.get("result")
        if isinstance(result, dict):
            leaf = result_path.parent
            for leg_index, leg in enumerate(result["legs"]):
                leg["name"] = f"renamed-leg-{result_index}-{leg_index}"
                old_full = leaf / leg["full_workload_mlir"]
                new_full_relative = f"relocated/{leg_index}.whole.mlir"
                new_full = leaf / new_full_relative
                new_full.parent.mkdir(parents=True, exist_ok=True)
                old_full.rename(new_full)
                leg["full_workload_mlir"] = new_full_relative
                for profile_index, profile in enumerate(leg["profiles"]):
                    profile["name"] = (
                        f"renamed-profile-{result_index}-{leg_index}-{profile_index}"
                    )
                    replacements = {
                        "profile": f"relocated/{profile_index}.evidence.log",
                        "compiled_profile_mlir": (
                            f"relocated/{profile_index}.physical.mlir"
                        ),
                        "compiled_gvml": f"relocated/{profile_index}.emitted.c",
                    }
                    for field, new_relative in replacements.items():
                        old_path = leaf / profile[field]
                        new_path = leaf / new_relative
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        old_path.rename(new_path)
                        profile[field] = new_relative
        _write_json(result_path, data)

    after = REPORTER.build_apu_v1_profile_calibration_report(tmp_path)

    assert after["content_addressing"] == before["content_addressing"]
    assert (
        after["selected_candidate_cross_shape_prediction"]
        == before["selected_candidate_cross_shape_prediction"]
    )
    for category in (
        REPORTER.DENSE_SEARCHED,
        REPORTER.SINGLETON_GEMV_BYPASS,
        REPORTER.REDUCTION_ONE_BYPASS,
    ):
        assert [
            record["historical_record_identity_fingerprint"]
            for record in after["profile_records"][category]
        ] == [
            record["historical_record_identity_fingerprint"]
            for record in before["profile_records"][category]
        ]
    assert after["report_fingerprint"] != before["report_fingerprint"]
    assert "renamed" not in json.dumps(after["content_addressing"])


def test_malformed_and_mixed_evidence_fail_closed(tmp_path):
    backend_root = tmp_path / "backend"
    backend_paths = _artifact_tree(backend_root)
    backend_data = json.loads(backend_paths["dense_first"].read_text(encoding="utf-8"))
    backend_data["backend"] = "apu-v2"
    _write_json(backend_paths["dense_first"], backend_data)
    with pytest.raises(REPORTER.ArtifactReportError, match="backend"):
        REPORTER.build_apu_v1_profile_calibration_report(backend_root)

    board_root = tmp_path / "board"
    board_paths = _artifact_tree(board_root)
    board_data = json.loads(board_paths["dense_first"].read_text(encoding="utf-8"))
    board_relative = board_data["result"]["legs"][0]["profiles"][0]["profile"]
    board_path = board_paths["dense_first"].parent / board_relative
    board_text = board_path.read_text(encoding="utf-8").replace(
        "correctness: PASS", "correctness: FAIL"
    )
    board_path.write_text(board_text, encoding="utf-8")
    with pytest.raises(REPORTER.ArtifactReportError, match="correctness"):
        REPORTER.build_apu_v1_profile_calibration_report(board_root)

    traversal_root = tmp_path / "traversal"
    traversal_paths = _artifact_tree(traversal_root)
    traversal_data = json.loads(
        traversal_paths["dense_first"].read_text(encoding="utf-8")
    )
    traversal_data["result"]["legs"][0]["profiles"][0]["profile"] = "../outside.log"
    _write_json(traversal_paths["dense_first"], traversal_data)
    with pytest.raises(REPORTER.ArtifactReportError, match="stay within"):
        REPORTER.build_apu_v1_profile_calibration_report(traversal_root)

    mixed_root = tmp_path / "mixed"
    mixed_paths = _artifact_tree(mixed_root)
    mixed_data = json.loads(mixed_paths["dense_duplicate"].read_text(encoding="utf-8"))
    candidates = mixed_data["result"]["legs"][0]["profiles"][0]["row_tile_selection"][
        "candidates"
    ]
    candidates[0]["estimated_shard_cycles"] += 1
    candidates[0]["estimated_composed_cycles"] += 2
    _write_json(mixed_paths["dense_duplicate"], mixed_data)
    with pytest.raises(REPORTER.ArtifactReportError, match="mixed analytical"):
        REPORTER.build_apu_v1_profile_calibration_report(mixed_root)


@pytest.mark.skipif(
    not REAL_CORPUS.is_dir(), reason="APUv1 artifact corpus unavailable"
)
def test_real_corpus_smoke_counts_and_metrics():
    report = REPORTER.build_apu_v1_profile_calibration_report(REAL_ARTIFACTS_ROOT)
    summary = report["artifact_summary"]

    assert summary["source_result_record_count"] == 30
    assert summary["measured_result_record_count"] == 15
    assert summary["not_applicable_result_record_count"] == 15
    assert summary["result_status_counts"] == {
        "NOT-APPLICABLE": 15,
        "PARTIAL": 11,
        "PASS": 4,
    }
    assert summary["profile_observation_count"] == 40
    assert summary["unique_profile_schedule_count"] == 25
    assert summary["profile_observation_counts_by_category"] == {
        REPORTER.DENSE_SEARCHED: 27,
        REPORTER.SINGLETON_GEMV_BYPASS: 12,
        REPORTER.REDUCTION_ONE_BYPASS: 1,
    }
    assert summary["unique_profile_schedule_counts_by_category"] == {
        REPORTER.DENSE_SEARCHED: 19,
        REPORTER.SINGLETON_GEMV_BYPASS: 5,
        REPORTER.REDUCTION_ONE_BYPASS: 1,
    }
    assert summary["unique_semantic_shape_counts_by_category"] == {
        REPORTER.DENSE_SEARCHED: 10,
        REPORTER.SINGLETON_GEMV_BYPASS: 5,
        REPORTER.REDUCTION_ONE_BYPASS: 1,
    }

    error = report["selected_candidate_cross_shape_prediction"]["error"]
    assert error["record_count"] == 19
    assert error["signed_relative_bias"] == pytest.approx(0.37829850699507706)
    assert error["median_absolute_percentage_error"] == pytest.approx(
        38.369447068026524
    )
    assert error["p90_absolute_percentage_error"] == pytest.approx(47.82686953191786)
    assert error["maximum_absolute_percentage_error"] == pytest.approx(
        48.91897220214234
    )
    coverage = report["candidate_domain_coverage"]
    assert coverage["candidate_domain_count"] == 19
    assert coverage["analytical_candidate_count"] == 76
    assert coverage["challenger_candidate_count"] == 57
    assert coverage["candidate_measurement_coverage"] == 0.25
