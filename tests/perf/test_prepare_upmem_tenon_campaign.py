# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests for the deterministic Tenon UPMEM campaign exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_upmem_tenon_campaign.py"
SPEC = importlib.util.spec_from_file_location("prepare_upmem_tenon_campaign", SCRIPT)
PREPARER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PREPARER
SPEC.loader.exec_module(PREPARER)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_array(path: Path, values, kind: str) -> dict[str, object]:
    values = tuple(values)
    formats = {"i32": "i", "u32": "I", "i64": "q"}
    data = struct.pack(f"<{len(values)}{formats[kind]}", *values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "dtype": kind,
        "elements": len(values),
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def _vec(n: int, mul: int, add: int, mod: int, bias: int) -> list[int]:
    return [((index * mul + add) % mod) - bias for index in range(n)]


def _matmul(lhs, rhs, rows: int, reduction: int, columns: int) -> list[int]:
    return [
        sum(
            lhs[row * reduction + k] * rhs[k * columns + column]
            for k in range(reduction)
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _trunc_div(numerator: int, denominator: int) -> int:
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def _round_closest(numerator: int, denominator: int) -> int:
    half = _trunc_div(denominator, 2)
    adjusted = numerator - half if numerator < 0 else numerator + half
    return _trunc_div(adjusted, denominator)


def _emit(root: Path, workload: str, arrays, spec) -> dict[str, object]:
    files = {
        f"{name}.{kind}.bin": _write_array(
            root / workload / f"{name}.{kind}.bin", values, kind
        )
        for name, (kind, values) in arrays.items()
    }
    return {"workload": workload, "spec": spec, "files": files}


def _canonical_inputs(root: Path) -> None:
    manifest = {}
    n = 12_288
    a = _vec(n, 17, 3, 31, 15)
    b = _vec(n, 7, 5, 29, 14)
    manifest["va"] = _emit(
        root,
        "va",
        {
            "a": ("i32", a),
            "b": ("i32", b),
            "expected": ("i32", [x + y for x, y in zip(a, b)]),
        },
        {"N": n},
    )
    manifest["red"] = _emit(
        root,
        "red",
        {"a": ("i32", a), "expected": ("i64", [sum(a)])},
        {"N": n},
    )
    manifest["geva"] = _emit(
        root,
        "geva",
        {
            "a": ("i32", a),
            "b": ("i32", b),
            "expected": ("i32", [2 * x - y for x, y in zip(a, b)]),
        },
        {"N": n, "alpha": 2, "beta": -1},
    )

    rows, columns = 96, 128
    matrix = _vec(rows * columns, 5, 1, 11, 5)
    vector = _vec(columns, 3, 2, 9, 4)
    mv = [
        sum(
            matrix[row * columns + column] * vector[column] for column in range(columns)
        )
        for row in range(rows)
    ]
    for workload, scale in (("mtv", 1), ("gemv", 2)):
        manifest[workload] = _emit(
            root,
            workload,
            {
                "a": ("i32", matrix),
                "x": ("i32", vector),
                "expected": ("i32", [scale * value for value in mv]),
            },
            {"M": rows, "K": columns, "alpha": scale},
        )

    batches, batch_rows, width = 12, 16, 32
    tensor = _vec(batches * batch_rows * width, 7, 2, 13, 6)
    shared_vector = _vec(width, 5, 1, 11, 5)
    ttv = [
        sum(
            tensor[row * width + column] * shared_vector[column]
            for column in range(width)
        )
        for row in range(batches * batch_rows)
    ]
    manifest["ttv"] = _emit(
        root,
        "ttv",
        {
            "a": ("i32", tensor),
            "x": ("i32", shared_vector),
            "expected": ("i32", ttv),
        },
        {"M": batches, "N": batch_rows, "K": width},
    )
    batch_vectors = _vec(batches * width, 9, 4, 17, 8)
    mmtv = [
        sum(
            tensor[(batch * batch_rows + row) * width + column]
            * batch_vectors[batch * width + column]
            for column in range(width)
        )
        for batch in range(batches)
        for row in range(batch_rows)
    ]
    manifest["mmtv"] = _emit(
        root,
        "mmtv",
        {
            "a": ("i32", tensor),
            "b": ("i32", batch_vectors),
            "expected": ("i32", mmtv),
        },
        {"M": batches, "N": batch_rows, "K": width},
    )

    hist_input = [(index * 37 + 11) % 4096 for index in range(n)]
    histogram = [0] * 128
    for value in hist_input:
        histogram[(value * 128) >> 12] += 1
    manifest["hist"] = _emit(
        root,
        "hist",
        {"input": ("i32", hist_input), "expected": ("u32", histogram)},
        {"N": n, "bins": 128, "depth": 12},
    )

    selection = _vec(n, 13, 7, 1001, 500)
    selected = [value for value in selection if value % 2]
    assert len(selected) == 6_224
    manifest["sel"] = _emit(
        root,
        "sel",
        {
            "input": ("i32", selection),
            "expected": ("i32", selected),
            "expected_count": ("u32", [len(selected)]),
        },
        {"N": n, "predicate": "odd"},
    )

    points, dimension, clusters = 120, 8, 4
    point_values = [
        ((point * 19 + feature * 7 + 3) % 101) - 50
        for point in range(points)
        for feature in range(dimension)
    ]
    centers = point_values[: clusters * dimension]
    sums = [[0] * dimension for _ in range(clusters)]
    counts = [0] * clusters
    for point in range(points):
        values = point_values[point * dimension : (point + 1) * dimension]
        chosen = min(
            range(clusters),
            key=lambda cluster: sum(
                (values[feature] - centers[cluster * dimension + feature]) ** 2
                for feature in range(dimension)
            ),
        )
        counts[chosen] += 1
        for feature, value in enumerate(values):
            sums[chosen][feature] += value
    updated = [
        (
            0
            if counts[cluster] == 0
            else _round_closest(sums[cluster][feature], counts[cluster])
        )
        for cluster in range(clusters)
        for feature in range(dimension)
    ]
    manifest["kmeans"] = _emit(
        root,
        "kmeans",
        {
            "points": ("i32", point_values),
            "initial_centroids": ("i32", centers),
            "expected_centroids": ("i32", updated),
            "expected_counts": ("u32", counts),
        },
        {"points": points, "dim": dimension, "k": clusters, "iters": 1},
    )

    samples, features = 120, 8
    x = [
        ((sample * 11 + feature * 5 + 1) % 17) - 8
        for sample in range(samples)
        for feature in range(features)
    ]
    linear_y = [((sample * 7 + 3) % 9) - 4 for sample in range(samples)]
    linear_gradient = [
        sum(
            (x[sample * features + feature] * (-(linear_y[sample] << 5))) >> 8
            for sample in range(samples)
        )
        for feature in range(features)
    ]
    packed_linear = [
        value
        for sample in range(samples)
        for value in x[sample * features : (sample + 1) * features] + [linear_y[sample]]
    ]
    manifest["linear_reg"] = _emit(
        root,
        "linear_reg",
        {
            "samples": ("i32", packed_linear),
            "expected_gradient": ("i64", linear_gradient),
        },
        {"samples": samples, "features": features, "shift": 5, "overflow_shift": 8},
    )
    logistic_y = [(sample * 5 + 1) & 1 for sample in range(samples)]
    logistic_gradient = [
        sum(
            x[sample * features + feature] * (1 - 2 * logistic_y[sample])
            for sample in range(samples)
        )
        for feature in range(features)
    ]
    packed_logistic = [
        value
        for sample in range(samples)
        for value in x[sample * features : (sample + 1) * features]
        + [logistic_y[sample]]
    ]
    manifest["logistic_reg"] = _emit(
        root,
        "logistic_reg",
        {
            "samples": ("i32", packed_logistic),
            "expected_twice_gradient_at_zero": ("i64", logistic_gradient),
        },
        {"samples": samples, "features": features},
    )

    mm_rows, mm_reduction, mm_columns = 12, 64, 128
    mm_lhs = _vec(mm_rows * mm_reduction, 7, 2, 13, 6)
    mm_rhs = _vec(mm_reduction * mm_columns, 5, 3, 11, 5)
    mm_output = _matmul(mm_lhs, mm_rhs, mm_rows, mm_reduction, mm_columns)
    for workload, repeats in (("1mm", 1), ("2mm", 2), ("3mm", 3)):
        manifest[workload] = _emit(
            root,
            workload,
            {
                "a": ("i32", mm_lhs),
                "b": ("i32", mm_rhs),
                "expected": ("i32", mm_output),
            },
            {"M": mm_rows, "K": mm_reduction, "N": mm_columns, "repeats": repeats},
        )

    patches, kernel_elements, filters = 192, 16, 32
    conv_lhs = _vec(patches * kernel_elements, 3, 1, 9, 4)
    conv_rhs = _vec(kernel_elements * filters, 7, 2, 13, 6)
    manifest["conv"] = _emit(
        root,
        "conv",
        {
            "patches": ("i32", conv_lhs),
            "filters": ("i32", conv_rhs),
            "expected": (
                "i32",
                _matmul(conv_lhs, conv_rhs, patches, kernel_elements, filters),
            ),
        },
        {"patches": patches, "kernel_elements": kernel_elements, "filters": filters},
    )
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _read_i32(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    return struct.unpack(f"<{len(data) // 4}i", data)


def test_complete_campaign_tree_is_deterministic(tmp_path):
    canonical = tmp_path / "inputs"
    _canonical_inputs(canonical)
    first, second = tmp_path / "first", tmp_path / "second"

    first_cases = PREPARER.prepare_campaign(canonical, first)
    second_cases = PREPARER.prepare_campaign(canonical, second)

    assert len(first_cases) == len(second_cases) == 16
    assert {case.name for case in first_cases} == set(PREPARER.WORKLOADS)
    assert _tree_hashes(first) == _tree_hashes(second)
    for workload in PREPARER.WORKLOADS:
        case = first / "programs" / workload
        assert {path.name for path in case.iterdir()} == {
            "source",
            "fixture",
            "provenance.json",
        }
        manifest = json.loads(
            (case / "fixture" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_dpus"] == 1
        assert manifest["num_tasklets"] == 12
        assert manifest["executions"]
        assert all(len(execution["dpus"]) == 1 for execution in manifest["executions"])


def test_representative_physical_layouts_invert_to_canonical_values(tmp_path):
    canonical = tmp_path / "inputs"
    _canonical_inputs(canonical)
    output = tmp_path / "campaign"
    PREPARER.prepare_campaign(canonical, output)

    va_heap = _read_i32(output / "programs" / "va" / "fixture" / "mram.input.bin")
    va_a = _read_i32(canonical / "va" / "a.i32.bin")
    va_b = _read_i32(canonical / "va" / "b.i32.bin")
    assert va_heap[:32] == va_a[:32]
    assert va_heap[32:64] == va_b[:32]

    mmtv_heap = _read_i32(output / "programs" / "mmtv" / "fixture" / "mram.input.bin")
    mmtv_a = _read_i32(canonical / "mmtv" / "a.i32.bin")
    mmtv_b = _read_i32(canonical / "mmtv" / "b.i32.bin")
    assert mmtv_heap[: 12 * 16 * 32] == mmtv_a
    vector_begin = 12 * 16 * 32
    assert mmtv_heap[vector_begin : vector_begin + 12 * 32] == mmtv_b
    assert set(mmtv_heap[vector_begin + 12 * 32 :]) == {0}
    assert not (
        output / "programs" / "mmtv" / "source" / "row-layout-search.json"
    ).exists()

    gemm_heap = _read_i32(output / "programs" / "1mm" / "fixture" / "mram.input.bin")
    gemm_a = _read_i32(canonical / "1mm" / "a.i32.bin")
    gemm_b = _read_i32(canonical / "1mm" / "b.i32.bin")
    row, column_block, reduction_block = 1, 2, 1
    packet = ((row * 8 + column_block) * 4 + reduction_block) * 272
    assert gemm_heap[packet : packet + 16] == gemm_a[80:96]
    for column_lane in range(16):
        column = column_block * 16 + column_lane
        begin = packet + 16 + column_lane * 16
        assert gemm_heap[begin : begin + 16] == tuple(
            gemm_b[k * 128 + column] for k in range(16, 32)
        )

    kmeans_heap = _read_i32(
        output / "programs" / "kmeans" / "fixture" / "mram.input.bin"
    )
    kmeans_points = _read_i32(canonical / "kmeans" / "points.i32.bin")
    kmeans_centroids = _read_i32(canonical / "kmeans" / "initial_centroids.i32.bin")
    assert kmeans_heap[:16] == kmeans_points[:8] + kmeans_centroids[:8]
    assert kmeans_heap[16:32] == kmeans_points[:8] + kmeans_centroids[8:16]


def test_offsets_sizes_repeats_and_hash_bindings(tmp_path):
    canonical = tmp_path / "inputs"
    _canonical_inputs(canonical)
    output = tmp_path / "campaign"
    PREPARER.prepare_campaign(canonical, output)

    for workload in PREPARER.WORKLOADS:
        case = output / "programs" / workload
        plan = json.loads(
            (case / "source" / "physical-plan.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (case / "fixture" / "manifest.json").read_text(encoding="utf-8")
        )
        provenance = json.loads((case / "provenance.json").read_text(encoding="utf-8"))
        outputs = sorted(
            plan["output_regions"].values(), key=lambda region: region["offset"]
        )
        output_offset = outputs[0]["offset"]
        output_bytes = outputs[-1]["offset"] + outputs[-1]["bytes"] - output_offset
        declaration = manifest["executions"][0]["dpus"][0]
        assert declaration["mram_heap_output"]["offset"] == output_offset
        assert (
            case / "fixture" / declaration["mram_heap_output"]["file"]
        ).stat().st_size == output_bytes
        assert (case / "fixture" / "mram.input.bin").stat().st_size == plan[
            "mram_image_bytes"
        ]
        source_bytes = (case / "source" / "task.c").read_bytes()
        assert _sha256(source_bytes) == plan["source_sha256"]
        assert provenance["dpu_source_sha256"] == _sha256(source_bytes)
        assert provenance["source_sha256"] == PREPARER.hash_tree(case / "source")
        assert provenance["fixture_sha256"] == PREPARER.hash_tree(case / "fixture")
        assert provenance["oracle_scope"] == PREPARER.ORACLE_SCOPE

    assert (
        len(
            json.loads(
                (output / "programs" / "1mm" / "fixture" / "manifest.json").read_text()
            )["executions"]
        )
        == 1
    )
    assert (
        len(
            json.loads(
                (output / "programs" / "2mm" / "fixture" / "manifest.json").read_text()
            )["executions"]
        )
        == 2
    )
    assert (
        len(
            json.loads(
                (output / "programs" / "3mm" / "fixture" / "manifest.json").read_text()
            )["executions"]
        )
        == 3
    )
    assert (
        output / "programs" / "sel" / "fixture" / "mram.output.expected.bin"
    ).stat().st_size == 49_152
    # Comparable k-means returns all 120x4 int64 distances for host postprocessing.
    assert (
        output / "programs" / "kmeans" / "fixture" / "mram.output.expected.bin"
    ).stat().st_size == 120 * 4 * 8


def test_calibrated_search_evidence_is_bound_to_every_modeled_plan(tmp_path):
    canonical = tmp_path / "inputs"
    _canonical_inputs(canonical)
    output = tmp_path / "campaign"
    PREPARER.prepare_campaign(canonical, output)

    modeled = set(PREPARER.PHYSICAL_SEARCH_WORKLOADS)
    assert modeled == {
        "va",
        "geva",
        "mtv",
        "gemv",
        "ttv",
        "mmtv",
        "sel",
        "1mm",
        "2mm",
        "3mm",
        "conv",
    }
    model_hashes = set()
    for workload in PREPARER.WORKLOADS:
        case = output / "programs" / workload
        source = case / "source"
        provenance = json.loads((case / "provenance.json").read_text(encoding="utf-8"))
        if workload not in modeled:
            assert not (source / "physical-search.json").exists()
            assert not (source / "physical-cost-model.json").exists()
            assert provenance["physical_search"]["status"] == "not_applicable"
            continue

        search_path = source / "physical-search.json"
        model_path = source / "physical-cost-model.json"
        plan_path = source / "physical-plan.json"
        task_path = source / "task.c"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        model = json.loads(model_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected = search["selected"]
        decision = selected["decision"]

        assert search["schema"] == "upmem-physical-search-v1"
        assert model["schema"] == "upmem-upimulator-physical-cost-v1"
        assert search["assignment_count"] == len(
            search["decision_space"]["assignments"]
        )
        assert [
            assignment["ordinal"]
            for assignment in search["decision_space"]["assignments"]
        ] == list(range(search["assignment_count"]))
        assert (
            selected["candidate_fingerprint"]
            == search["ranked"][0]["candidate_fingerprint"]
        )
        assert decision == search["ranked"][0]["decision"]
        assert search["ranked"][0]["plan_materialized"] is True
        expected_status = (
            "calibrated_decision_stratum_shape_extrapolation"
            if workload in {"geva", "ttv", "conv"}
            else "calibrated_exact_problem_and_decision_stratum"
        )
        assert selected["calibration_status"] == expected_status

        model_sha256 = _sha256(model_path.read_bytes())
        search_sha256 = _sha256(search_path.read_bytes())
        model_hashes.add(model_sha256)
        assert search["model_fingerprint"] == model["model_fingerprint"]
        assert search["cost_model_binding"] == {
            "file": "physical-cost-model.json",
            "model_fingerprint": model["model_fingerprint"],
            "sha256": model_sha256,
        }
        assert search["emitted_plan_binding"]["physical_plan_sha256"] == (
            _sha256(plan_path.read_bytes())
        )
        assert search["emitted_plan_binding"]["device_source_sha256"] == (
            _sha256(task_path.read_bytes())
        )
        recorded = provenance["physical_search"]
        assert recorded["status"] == "selected_and_bound"
        assert recorded["search_sha256"] == search_sha256
        assert recorded["cost_model_sha256"] == model_sha256
        assert recorded["search_fingerprint"] == search["search_fingerprint"]
        assert (
            recorded["selected_candidate_fingerprint"]
            == selected["candidate_fingerprint"]
        )
        assert (
            recorded["selected_decision_fingerprint"]
            == selected["decision_fingerprint"]
        )

        assert decision["tasklets"] == plan["num_tasklets"] == 12
        assert decision["ownership"] == "contiguous"
        if workload in {"va", "geva"}:
            assert decision["layout"] == "fused_replicated"
            assert decision["dma_bytes"] == plan["fused_dma_bytes"] == 256
            assert decision["chunk_elements"] == plan["operand_dma_bytes"] // 4
        elif workload in {"mtv", "gemv", "ttv"}:
            assert decision["layout"] == "fused_replicated"
            assert decision["vector_residency"] == "fused_dma_packet"
            assert decision["dma_bytes"] == plan["packet_bytes"] == 128
            assert decision["chunk_elements"] == plan["operand_dma_bytes"] // 4
            assert search["problem"]["rows"] == plan["flattened_rows"]
            assert search["problem"]["batches"] == plan["shape"][0] == 1
            assert search["problem"]["reduction"] == plan["shape"][2]
        elif workload == "mmtv":
            assert decision["layout"] == plan["layout"] == "separate"
            assert (
                decision["vector_residency"]
                == (plan["vector_mode"])
                == "tasklet_private_wram"
            )
            assert decision["dma_bytes"] == plan["operand_dma_bytes"] == 64
            assert decision["chunk_elements"] == plan["matrix_chunk_elements"] == 16
            assert search["problem"]["rows"] == plan["flattened_rows"] == 192
            assert search["problem"]["batches"] == plan["shape"][0] == 12
            assert search["problem"]["reduction"] == plan["shape"][2] == 32
            assert search["assignment_count"] == 3
            assert selected["predicted_logic_cycles"] == 211_371
            assert selected["calibration_evidence_ids"]
        elif workload in {"1mm", "2mm", "3mm", "conv"}:
            assert decision["layout"] == "fused_replicated"
            assert decision["rhs_residency"] == "fused_dma_packet"
            assert decision["dma_bytes"] == plan["packet_bytes"] == 1088
            assert decision["nc"] == plan["column_tile"] == 16
            assert decision["kc"] == plan["reduction_tile"] == 16
        else:
            assert workload == "sel"
            assert decision["layout"] == "separate"
            assert decision["predicate_lowering"] == "conditional_zero"
            assert decision["dma_bytes"] == 4 * plan["dma_elements"] == 64
            assert decision["chunk_elements"] == plan["dma_elements"] == 16

    assert len(model_hashes) == 1


def test_mmtv_resident_vector_plan_supersedes_content_aware_row_layout(tmp_path):
    canonical = tmp_path / "inputs"
    _canonical_inputs(canonical)
    output = tmp_path / "campaign"
    PREPARER.prepare_campaign(canonical, output)

    case = output / "programs" / "mmtv"
    search_path = case / "source" / "row-layout-search.json"
    plan_path = case / "source" / "physical-plan.json"
    task_path = case / "source" / "task.c"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = task_path.read_text(encoding="utf-8")
    provenance = json.loads((case / "provenance.json").read_text(encoding="utf-8"))

    assert not search_path.exists()
    assert plan["kind"] == "upmem-batch-resident-int32-matrix-vector"
    assert plan["vector_mode"] == "tasklet_private_wram"
    assert plan["row_ownership"] == "tasklet-id-is-batch-id"
    assert plan["row_layout"]["identity"] is True
    assert plan["row_layout"]["physical_to_logical_rows"] == list(range(192))
    assert set(plan["mram_regions"]) == {"matrix", "vector", "output"}
    assert plan["mram_regions"]["matrix"]["offset"] == 0
    assert plan["mram_regions"]["matrix"]["bytes"] == 24_576
    assert plan["mram_regions"]["vector"]["offset"] == 24_576
    assert plan["mram_regions"]["vector"]["bytes"] == 1_536
    assert plan["mram_regions"]["output"]["offset"] == 26_112
    assert plan["mram_regions"]["output"]["bytes"] == 768
    assert "mem_reset();" in source
    assert source.count("mem_alloc(") == 3
    assert source.index("&vectors[tid * TENON_COLUMNS") < source.index(
        "for (uint32_t row = 0; row < TENON_ROWS_PER_BATCH; ++row)"
    )
    assert "block * TENON_CHUNK_ELEMENTS" in source
    assert "TENON_OUTPUT_BYTES" in source

    fixture_input = case / "fixture" / "mram.input.bin"
    fixture_output = case / "fixture" / "mram.output.expected.bin"
    physical_input = _read_i32(fixture_input)
    matrix = _read_i32(canonical / "mmtv" / "a.i32.bin")
    vectors = _read_i32(canonical / "mmtv" / "b.i32.bin")
    assert physical_input[: len(matrix)] == matrix
    assert physical_input[len(matrix) : len(matrix) + len(vectors)] == vectors
    logical_expected = _read_i32(canonical / "mmtv" / "expected.i32.bin")
    physical_expected = _read_i32(fixture_output)
    assert physical_expected == logical_expected

    recorded = provenance["row_layout_search"]
    assert recorded["status"] == "not_applicable"
    assert "superseded" in recorded["reason"]
    assert recorded["selected_vector_mode"] == "tasklet_private_wram"
    assert recorded["identity_row_layout"] is True
    assert recorded["physical_plan_sha256"] == _sha256(plan_path.read_bytes())
    assert recorded["device_source_sha256"] == _sha256(task_path.read_bytes())
    assert recorded["fixture_input_sha256"] == _sha256(fixture_input.read_bytes())
    assert recorded["fixture_output_sha256"] == _sha256(fixture_output.read_bytes())
    assert "superseded" in provenance["notes"]["row_layout_qualification"]
    assert "private WRAM" in provenance["notes"]["resident_vector_qualification"]

    for workload in set(PREPARER.WORKLOADS) - {"mmtv"}:
        other = output / "programs" / workload
        assert not (other / "source" / "row-layout-search.json").exists()
        other_provenance = json.loads(
            (other / "provenance.json").read_text(encoding="utf-8")
        )
        assert other_provenance["row_layout_search"]["status"] == "not_applicable"

    # The post-pack binding is fail-closed: a single corrupted matrix byte must
    # not retain the superseded row-layout qualification.
    corrupted = bytearray(fixture_input.read_bytes())
    corrupted[0] ^= 1
    fixture_input.write_bytes(corrupted)
    canonical_access = PREPARER.CanonicalInputs(canonical)
    resident_plan = PREPARER.UPMEMMatrixVectorPlan(
        rows=16,
        columns=32,
        batches=12,
        vector_mode="tasklet_private_wram",
    )
    with pytest.raises(ValueError, match="matrix differs from fixture packing"):
        PREPARER._write_mmtv_row_layout_evidence(
            case / "source",
            case / "fixture",
            canonical_access,
            "mmtv",
            resident_plan,
            task_path,
            plan_path,
        )
