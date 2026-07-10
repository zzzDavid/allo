import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pim_schedule_nonregression.py"
MANIFEST = ROOT / "tests" / "perf" / "data" / "pim_schedule_incumbents.json"
SPEC = importlib.util.spec_from_file_location("pim_schedule_nonregression", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def _upmem_result(*, cycles=100, semantic_status="FULL"):
    partial = semantic_status == "PARTIAL"
    return {
        "kernel": "gemm",
        "dataset": "LARGE_DATASET",
        "dtype": "int32",
        "dimensions": {"P": 10, "Q": 12, "R": 11},
        "description": "dense product",
        "status": "measured_composition",
        "semantic_status": semantic_status,
        "omitted_semantics": "beta epilogue" if partial else None,
        "metric": "uPIMulator_logic_cycles",
        "cycles": cycles,
        "correctness": "simulator_checker_clean_finish_for_each_primitive",
        "provenance": {"allo_commit": "1" * 40},
    }


def _apu_v1_result(runs, *, warmup=True):
    result = {
        "metric": "device-compute microseconds",
        "cycles": runs[0]["cycles"],
        "runs": runs,
        "execution_scope": "single APUC",
        "correctness": "PASS bit-exact modulo 2^16",
        "legs": [
            {
                "name": "gemm",
                "shape": {"M": 10, "N": 11, "K": 12},
                "operation_count": 1,
                "profiles": [
                    {
                        "name": "gemm",
                        "profile_shape": {"M": 10, "N": 11, "K": 12},
                        "row_waves": 1,
                        "plan": "temporal_dma",
                        "compiled_profile_mlir": "compiled/gemm.profile.mlir",
                        "compiled_gvml": "compiled/gemm.gvml.c",
                    }
                ],
            }
        ],
    }
    if warmup:
        result["warmup"] = {"count": 1, "discarded": True}
    return {
        "kernel": "gemm",
        "backend": "apu-v1",
        "framework": "tenon",
        "dataset": "PolyBench/C 4.2.1 LARGE",
        "dtype": "uint16 (16-bit modular integer datapath)",
        "status": "PASS",
        "detail": "real-device full LARGE contraction composition",
        "result": result,
        "tenon_revision": "2" * 40,
    }


def _apu_v2_result(runs, *, warmup=True):
    result = {
        "metric": "measured persistent-stage device ticks",
        "cycles": runs[0]["cycles"],
        "wall_metric": "measured host wall: h2d + host task dispatch + d2h",
        "wall_microseconds": runs[0]["wall_microseconds"],
        "correctness": "PASS bit-exact modulo 2^16",
        "profiles": [
            {
                "kind": "persistent_column_batched_gemm",
                "batch_columns": 31,
                "profile": "profile/gemm.board.log",
            }
        ],
        "runs": runs,
    }
    if warmup:
        result["warmup"] = {"count": 1, "discarded": True}
    return {
        "kernel": "gemm",
        "backend": "apu-v2",
        "framework": "tenon",
        "dataset": "PolyBench/C 4.2.1 LARGE",
        "dtype": "uint16 (16-bit modular integer datapath)",
        "status": "MEASURED-COMPOSITION",
        "semantic_status": "FULL uint16 specialization",
        "shape": {"P": 10, "Q": 12, "R": 11},
        "result": result,
        "tenon_revision": "3" * 40,
    }


def _fixture(backend):
    if backend == "samsung-hbm-pim":
        result = {
            "kernel": "gemm",
            "target": backend,
            "dataset": "LARGE_DATASET",
            "dtype": "float16",
            "status": "PASS",
            "correctness": "PASS",
            "metric": "PIMSimulator cycles (1 ns/cycle)",
            "cycles": 100,
            "legs": [
                {
                    "name": "columns",
                    "operation": "gemv",
                    "logical_shape": {"rows": 10, "reduction": 12},
                    "tenon_physical_shape": {"rows": 16, "reduction": 16},
                    "count": 11,
                    "compiled_crf": "compiled/columns.crf",
                }
            ],
            "tenon_commit": "4" * 40,
        }
        files = {
            "compiled/columns.crf": b"MAC src1=EVEN_BANK\nMAC src1=ODD_BANK\n"
        }
    elif backend == "sk-hynix-aim":
        result = {
            "benchmark": "gemm",
            "backend": backend,
            "framework": "tenon",
            "dataset": "LARGE_DATASET",
            "dtype": "bf16",
            "shapes": {"P": 10, "Q": 12, "R": 11},
            "status": "PASS",
            "detail": "complete batched-GEMV decomposition",
            "performance": {
                "metric": "device-compute us",
                "cycles": 100,
            },
            "provenance": {"tenon_commit": "5" * 40},
        }
        files = {
            "compiled/aim.trace": b"# TENON_GEMV\nAiM MAC_ABK 0 1\n",
            "workload_spec.json": _json_bytes(
                {"mapping": "SPMW mapping=[32]; MAC_ABK spans 16 banks/channel"}
            ),
        }
    elif backend == "upmem":
        result = _upmem_result()
        files = {
            "compiled/dense_tile.json": _json_bytes(
                {
                    "kind": "upmem-dense-int32-tile",
                    "shape": [2, 2, 12],
                    "reduction_tile": 4,
                    "num_tasklets": 2,
                }
            ),
            "compiled/dense_tile.dpu.c": b"int main(void) { return 0; }\n",
        }
    elif backend == "apu-v1":
        result = _apu_v1_result([{"cycles": 100}])
        files = {
            "compiled/gemm.profile.mlir": b"module { func.func @gemm() }\n",
            "compiled/gemm.gvml.c": b"void gemm(void) {}\n",
        }
    else:
        result = _apu_v2_result([{"cycles": 100, "wall_microseconds": 1000}])
        files = {
            "compiled/device_sources/CMakeLists.txt": b"add_subdirectory(device)\n",
            "compiled/device_sources/apu_g2_params.h": b"#define BATCH_COLUMNS 31\n",
            "compiled/device_sources/device/CMakeLists.txt": b"add_library(device)\n",
            "compiled/device_sources/device/apu_g2_u16_gemm.cc": (
                b"void gemm_device(void) {}\n"
            ),
            "compiled/device_sources/device/gsi_library.h": b"void gsi_entry(void);\n",
            "compiled/device_sources/host.cc": b"int main(void) { return 0; }\n",
            "compiled/execution_graph.json": _json_bytes(
                {
                    "metadata": {
                        "program": "column_batched_gemm_u16",
                        "transport_schedule": {
                            "kind": "column_batched_u16_gemm",
                            "batch_columns": 31,
                            "reduction_tile": 128,
                        },
                    }
                }
            ),
            "compiled/recipe.json": _json_bytes({"kind": "column_batched_u16_gemm"}),
            "compiled/tenon_sources.json": _json_bytes({"generated": ["host.cc"]}),
            "compiled/workload.mlir": b"module { func.func @gemm() }\n",
            "profile/gemm.board.log": b"measured profile\n",
            "profile/gemm.json": _json_bytes(
                {
                    "cycles": 100,
                    "extra": {
                        "schedule": {
                            "kind": "column_batched_u16_gemm",
                            "batch_columns": 31,
                            "reduction_tile": 128,
                        }
                    },
                }
            ),
        }
    return result, files


def _metrics(backend):
    if backend in {"samsung-hbm-pim", "sk-hynix-aim", "upmem"}:
        return [
            {
                "kind": "cycles",
                "baseline": 100,
                "evidence": "cycle_simulator",
                "policy": "cycle_simulator",
            }
        ]
    if backend == "apu-v1":
        return [
            {
                "kind": "cycles",
                "baseline": 100,
                "evidence": "hardware_cycles",
                "policy": "apu_v1_cycles",
            }
        ]
    return [
        {
            "kind": "device_ticks",
            "baseline": 100,
            "evidence": "hardware_cycles",
            "policy": "apu_v2_device_ticks",
        },
        {
            "kind": "wall_us",
            "baseline": 1000,
            "evidence": "hardware_wall",
            "policy": "apu_v2_wall_us",
        },
    ]


def _write_case(root, backend, result, files):
    leaf = root / "polybench" / backend / "tenon" / "gemm"
    leaf.mkdir(parents=True)
    result_path = leaf / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    for relative, content in files.items():
        path = leaf / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return result_path


def _entry(tmp_path, backend, baseline, files):
    root = tmp_path / "incumbent"
    result_path = _write_case(root, backend, baseline, files)
    normalized = CHECKER.normalize_result(backend, baseline)
    schedule = CHECKER.normalize_schedule(backend, baseline, result_path)
    result_relative = PurePosixPath("polybench", backend, "tenon", "gemm", "result.json")
    leaf_relative = result_relative.parent
    artifact_paths = set(CHECKER.schedule_artifact_paths(backend, schedule))
    if backend == "apu-v2":
        artifact_paths.update(
            relative
            for relative in files
            if PurePosixPath(relative).is_relative_to(PurePosixPath("compiled"))
            and PurePosixPath(relative).name != "tenon_sources.json"
        )
    artifacts = [
        {
            "path": (leaf_relative / relative).as_posix(),
            "sha256": CHECKER.sha256_file(result_path.parent / relative),
        }
        for relative in sorted(artifact_paths)
    ]
    return {
        "key": f"{backend}/gemm",
        "result_path": result_relative.as_posix(),
        "dataset": normalized.dataset,
        "dtype": normalized.dtype,
        "shape": normalized.shape,
        "semantic": dict(normalized.semantic),
        "correctness": "pass",
        "metrics": _metrics(backend),
        "producer_revision": normalized.producer_revision,
        "schedule": schedule,
        "schedule_artifacts": artifacts,
    }


def _manifest(entry):
    return {
        "schema_version": 1,
        "source": {
            "artifact_revision": "0" * 40,
            "worktree_clean": True,
        },
        "policies": CHECKER.policy_template(),
        "record_count": 1,
        "records": [entry],
    }


def _check(root, entry, backend, candidate, files, overrides=None, **kwargs):
    candidate_files = dict(files)
    for relative, content in (overrides or {}).items():
        if content is None:
            candidate_files.pop(relative, None)
        else:
            candidate_files[relative] = content
    _write_case(root, backend, candidate, candidate_files)
    return CHECKER.check_artifacts(_manifest(entry), root, **kwargs)


def _codes(report):
    return {finding.code for finding in report.findings}


def test_exact_simulator_policy_passes_equal_and_fails_one_cycle(tmp_path):
    baseline, files = _fixture("upmem")
    entry = _entry(tmp_path, "upmem", baseline, files)
    assert _check(tmp_path / "equal", entry, "upmem", baseline, files).ok
    slower = copy.deepcopy(baseline)
    slower["cycles"] = 101
    assert "performance_regression" in _codes(
        _check(tmp_path / "slower", entry, "upmem", slower, files)
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("dataset", "SMALL_DATASET", "dataset_mismatch"),
        ("dtype", "int16", "dtype_mismatch"),
        ("dimensions", {"P": 9, "Q": 12, "R": 11}, "shape_mismatch"),
        ("correctness", "FAIL", "correctness_downgrade"),
    ],
)
def test_metadata_and_correctness_downgrades_fail(tmp_path, field, value, code):
    baseline, files = _fixture("upmem")
    entry = _entry(tmp_path, "upmem", baseline, files)
    candidate = copy.deepcopy(baseline)
    candidate[field] = value
    assert code in _codes(_check(tmp_path / "candidate", entry, "upmem", candidate, files))


def test_partial_full_and_omitted_semantics_never_cross_compare(tmp_path):
    full, files = _fixture("upmem")
    full_entry = _entry(tmp_path / "full", "upmem", full, files)
    partial = _upmem_result(semantic_status="PARTIAL")
    assert "semantic_status_mismatch" in _codes(
        _check(tmp_path / "full_to_partial", full_entry, "upmem", partial, files)
    )
    partial_entry = _entry(tmp_path / "partial", "upmem", partial, files)
    assert "semantic_status_mismatch" in _codes(
        _check(tmp_path / "partial_to_full", partial_entry, "upmem", full, files)
    )
    expanded = copy.deepcopy(partial)
    expanded["omitted_semantics"] = "alpha and beta epilogues"
    assert "omitted_semantics_expanded" in _codes(
        _check(tmp_path / "expanded", partial_entry, "upmem", expanded, files)
    )


def test_metric_kind_mismatch_fails(tmp_path):
    baseline, files = _fixture("upmem")
    entry = _entry(tmp_path, "upmem", baseline, files)
    candidate = copy.deepcopy(baseline)
    candidate["metric"] = "host wall microseconds"
    assert "metric_mismatch" in _codes(
        _check(tmp_path / "candidate", entry, "upmem", candidate, files)
    )


def test_apu_v1_five_measured_runs_and_discarded_warmup_gate(tmp_path):
    baseline, files = _fixture("apu-v1")
    entry = _entry(tmp_path, "apu-v1", baseline, files)
    four = _apu_v1_result([{"cycles": 101}] * 4)
    assert "insufficient_repetitions" in _codes(
        _check(tmp_path / "four", entry, "apu-v1", four, files)
    )
    boundary = _apu_v1_result([{"cycles": 101}] * 5)
    assert _check(tmp_path / "boundary", entry, "apu-v1", boundary, files).ok
    no_warmup = _apu_v1_result([{"cycles": 101}] * 5, warmup=False)
    no_warmup_report = _check(
        tmp_path / "no_warmup", entry, "apu-v1", no_warmup, files
    )
    assert "insufficient_warmup" in _codes(no_warmup_report)
    regression = _apu_v1_result([{"cycles": 102}] * 5)
    assert "performance_regression" in _codes(
        _check(tmp_path / "regression", entry, "apu-v1", regression, files)
    )


def test_apu_v2_seven_measured_runs_and_discarded_warmup_gate(tmp_path):
    baseline, files = _fixture("apu-v2")
    entry = _entry(tmp_path, "apu-v2", baseline, files)
    six = _apu_v2_result([{"cycles": 101, "wall_microseconds": 1030}] * 6)
    six_report = _check(tmp_path / "six", entry, "apu-v2", six, files)
    assert len([item for item in six_report.insufficient if item.code == "insufficient_repetitions"]) == 2
    boundary = _apu_v2_result([{"cycles": 101, "wall_microseconds": 1030}] * 7)
    assert _check(tmp_path / "boundary", entry, "apu-v2", boundary, files).ok
    no_warmup = _apu_v2_result(
        [{"cycles": 101, "wall_microseconds": 1030}] * 7,
        warmup=False,
    )
    assert "insufficient_warmup" in _codes(
        _check(tmp_path / "no_warmup", entry, "apu-v2", no_warmup, files)
    )
    device_regression = _apu_v2_result(
        [{"cycles": 102, "wall_microseconds": 1000}] * 7
    )
    assert "performance_regression" in _codes(
        _check(tmp_path / "device", entry, "apu-v2", device_regression, files)
    )
    wall_median_regression = _apu_v2_result(
        [{"cycles": 100, "wall_microseconds": 1031}] * 7
    )
    assert "performance_regression" in _codes(
        _check(tmp_path / "wall_median", entry, "apu-v2", wall_median_regression, files)
    )
    wall_outlier_runs = [{"cycles": 100, "wall_microseconds": 1000}] * 6
    wall_outlier_runs.append({"cycles": 100, "wall_microseconds": 1051})
    assert "performance_regression" in _codes(
        _check(
            tmp_path / "wall_outlier",
            entry,
            "apu-v2",
            _apu_v2_result(wall_outlier_runs),
            files,
        )
    )


@pytest.mark.parametrize(
    "backend",
    ["samsung-hbm-pim", "sk-hynix-aim", "upmem", "apu-v1", "apu-v2"],
)
def test_schedule_descriptor_mismatch_for_every_backend(tmp_path, backend):
    baseline, files = _fixture(backend)
    entry = _entry(tmp_path, backend, baseline, files)
    candidate = copy.deepcopy(baseline)
    overrides = {}
    if backend == "samsung-hbm-pim":
        candidate["legs"][0]["tenon_physical_shape"]["reduction"] = 32
    elif backend == "sk-hynix-aim":
        overrides["workload_spec.json"] = _json_bytes(
            {"mapping": "SPMW mapping=[16]; MAC_ABK spans 16 banks/channel"}
        )
    elif backend == "upmem":
        tile = json.loads(files["compiled/dense_tile.json"])
        tile["reduction_tile"] = 8
        overrides["compiled/dense_tile.json"] = _json_bytes(tile)
    elif backend == "apu-v1":
        candidate["result"]["legs"][0]["profiles"][0]["plan"] = "different_plan"
    else:
        candidate["result"]["profiles"][0]["batch_columns"] = 30
    assert "schedule_mismatch" in _codes(
        _check(
            tmp_path / "candidate",
            entry,
            backend,
            candidate,
            files,
            overrides,
        )
    )


def test_apu_v1_row_tile_and_apu_v2_measurement_and_graph_evidence(tmp_path):
    apu_v1, v1_files = _fixture("apu-v1")
    v1_entry = _entry(tmp_path / "v1", "apu-v1", apu_v1, v1_files)
    changed_row_tile = copy.deepcopy(apu_v1)
    changed_row_tile["result"]["legs"][0]["profiles"][0]["profile_shape"]["M"] = 5
    assert "schedule_mismatch" in _codes(
        _check(
            tmp_path / "v1_candidate",
            v1_entry,
            "apu-v1",
            changed_row_tile,
            v1_files,
        )
    )

    apu_v2, v2_files = _fixture("apu-v2")
    v2_entry = _entry(tmp_path / "v2", "apu-v2", apu_v2, v2_files)
    measured_rerun = _apu_v2_result(
        [{"cycles": 100, "wall_microseconds": 1000}] * 7
    )
    profile_record = json.loads(v2_files["profile/gemm.json"])
    profile_record["measurement_note"] = "fresh rerun"
    changed_profile = _check(
        tmp_path / "v2_changed_log",
        v2_entry,
        "apu-v2",
        measured_rerun,
        v2_files,
        {
            "profile/gemm.board.log": b"different rerun measurements\n",
            "profile/gemm.json": _json_bytes(profile_record),
        },
    )
    assert changed_profile.ok
    for field, value in (("reduction_tile", 64), ("kind", "different_transport")):
        graph = json.loads(v2_files["compiled/execution_graph.json"])
        graph["metadata"]["transport_schedule"][field] = value
        report = _check(
            tmp_path / f"v2_{field}",
            v2_entry,
            "apu-v2",
            apu_v2,
            v2_files,
            {"compiled/execution_graph.json": _json_bytes(graph)},
        )
        assert "schedule_mismatch" in _codes(report)
        assert "schedule_artifact_mismatch" in _codes(report)


@pytest.mark.parametrize("replacement", [b"changed code\n", None])
def test_schedule_artifact_byte_mutation_and_missing_file_fail(tmp_path, replacement):
    baseline, files = _fixture("upmem")
    entry = _entry(tmp_path, "upmem", baseline, files)
    report = _check(
        tmp_path / "candidate",
        entry,
        "upmem",
        baseline,
        files,
        {"compiled/dense_tile.dpu.c": replacement},
    )
    assert "schedule_artifact_mismatch" in _codes(report)


@pytest.mark.parametrize(
    "relative",
    [
        "compiled/execution_graph.json",
        "compiled/recipe.json",
        "compiled/workload.mlir",
        "compiled/device_sources/CMakeLists.txt",
        "compiled/device_sources/apu_g2_params.h",
        "compiled/device_sources/host.cc",
        "compiled/device_sources/device/apu_g2_u16_gemm.cc",
    ],
)
@pytest.mark.parametrize("replacement", [b"changed artifact\n", None])
def test_apu_v2_immutable_artifact_mutation_and_deletion_fail(
    tmp_path, relative, replacement
):
    baseline, files = _fixture("apu-v2")
    entry = _entry(tmp_path, "apu-v2", baseline, files)
    report = _check(
        tmp_path / "candidate",
        entry,
        "apu-v2",
        baseline,
        files,
        {relative: replacement},
    )
    assert "schedule_artifact_mismatch" in _codes(report)


def test_manifest_constrains_apu_v2_extras_without_relaxing_other_backends(tmp_path):
    baseline, files = _fixture("apu-v2")
    entry = _entry(tmp_path / "v2", "apu-v2", baseline, files)
    missing_graph = copy.deepcopy(entry)
    missing_graph["schedule_artifacts"] = [
        artifact
        for artifact in missing_graph["schedule_artifacts"]
        if not artifact["path"].endswith("/compiled/execution_graph.json")
    ]
    with pytest.raises(CHECKER.ManifestError, match="do not cover the schedule"):
        CHECKER.validate_manifest(_manifest(missing_graph))

    outside_compiled = copy.deepcopy(entry)
    outside_compiled["schedule_artifacts"].append(
        {
            "path": "polybench/apu-v2/tenon/gemm/profile/gemm.json",
            "sha256": "0" * 64,
        }
    )
    with pytest.raises(CHECKER.ManifestError, match="compiled directory"):
        CHECKER.validate_manifest(_manifest(outside_compiled))

    inventory = copy.deepcopy(entry)
    inventory["schedule_artifacts"].append(
        {
            "path": "polybench/apu-v2/tenon/gemm/compiled/tenon_sources.json",
            "sha256": "0" * 64,
        }
    )
    with pytest.raises(CHECKER.ManifestError, match="tenon_sources.json"):
        CHECKER.validate_manifest(_manifest(inventory))

    upmem, upmem_files = _fixture("upmem")
    upmem_entry = _entry(tmp_path / "upmem", "upmem", upmem, upmem_files)
    upmem_entry["schedule_artifacts"].append(
        {
            "path": "polybench/upmem/tenon/gemm/compiled/extra.c",
            "sha256": "0" * 64,
        }
    )
    with pytest.raises(CHECKER.ManifestError, match="do not cover the schedule"):
        CHECKER.validate_manifest(_manifest(upmem_entry))


def test_baseline_mode_allows_historical_single_hardware_sample(tmp_path):
    baseline, files = _fixture("apu-v2")
    baseline["result"].pop("warmup")
    entry = _entry(tmp_path, "apu-v2", baseline, files)
    report = _check(
        tmp_path / "candidate",
        entry,
        "apu-v2",
        baseline,
        files,
        validate_baseline=True,
    )
    assert report.ok


def test_checked_in_manifest_schema_count_policies_and_hashes_without_checkout():
    manifest = CHECKER.load_manifest(MANIFEST)
    assert manifest["record_count"] == 76
    assert manifest["source"]["artifact_revision"] == (
        "e51dcf64fde1e7383a55fbdb1040c0b012b0e7ee"
    )
    assert manifest["policies"]["apu_v1_cycles"]["min_samples"] == 5
    assert manifest["policies"]["apu_v2_device_ticks"]["min_samples"] == 7
    assert manifest["policies"]["apu_v2_wall_us"]["min_samples"] == 7
    assert all(
        manifest["policies"][name]["warmup_samples"] == 1
        for name in ("apu_v1_cycles", "apu_v2_device_ticks", "apu_v2_wall_us")
    )
    counts = Counter(record["key"].split("/", 1)[0] for record in manifest["records"])
    assert counts == {
        "samsung-hbm-pim": 15,
        "sk-hynix-aim": 16,
        "upmem": 15,
        "apu-v1": 15,
        "apu-v2": 15,
    }
    apu_v2_artifacts = []
    for record in manifest["records"]:
        backend = record["key"].split("/", 1)[0]
        assert record["schedule"]
        assert record["schedule_artifacts"]
        for artifact in record["schedule_artifacts"]:
            path = PurePosixPath(artifact["path"])
            assert not path.is_absolute() and ".." not in path.parts
            assert len(artifact["sha256"]) == 64
        suffixes = {PurePosixPath(item["path"]).suffix for item in record["schedule_artifacts"]}
        if backend == "samsung-hbm-pim":
            assert ".crf" in suffixes
        elif backend == "sk-hynix-aim":
            assert ".trace" in suffixes
        elif backend == "upmem":
            assert {".json", ".c"} <= suffixes
        elif backend == "apu-v1":
            assert {".mlir", ".c"} <= suffixes
        else:
            workload = record["key"].split("/", 1)[1]
            expected_count = 36 if workload == "gesummv" else 37
            assert len(record["schedule_artifacts"]) == expected_count
            compiled = PurePosixPath(record["result_path"]).parent / "compiled"
            paths = [
                PurePosixPath(artifact["path"])
                for artifact in record["schedule_artifacts"]
            ]
            assert all(path.is_relative_to(compiled) for path in paths)
            assert compiled / "execution_graph.json" in paths
            apu_v2_artifacts.extend(paths)

    assert len(apu_v2_artifacts) == 554
    categories = Counter()
    for path in apu_v2_artifacts:
        compiled_index = path.parts.index("compiled")
        relative = PurePosixPath(*path.parts[compiled_index + 1 :])
        if relative == PurePosixPath("execution_graph.json"):
            categories["execution_graph"] += 1
        elif relative == PurePosixPath("recipe.json"):
            categories["recipe_json"] += 1
        elif relative == PurePosixPath("workload.mlir"):
            categories["workload_mlir"] += 1
        elif relative.name == "CMakeLists.txt":
            categories["cmake"] += 1
        elif relative.suffix == ".h":
            categories["header"] += 1
        elif relative.suffix == ".cc" and relative.parent.name == "device":
            categories["device_cc"] += 1
        elif relative.suffix == ".cc":
            categories["host_cc"] += 1
        else:
            categories["unexpected"] += 1
        assert "profile" not in path.parts
        assert path.name not in {
            "result.json",
            "workload.py",
            "tenon_sources.json",
        }
        assert path.suffix not in {".a", ".bin", ".elf", ".o", ".so"}
    assert categories == {
        "host_cc": 225,
        "device_cc": 225,
        "cmake": 30,
        "header": 30,
        "recipe_json": 15,
        "workload_mlir": 14,
        "execution_graph": 15,
    }
