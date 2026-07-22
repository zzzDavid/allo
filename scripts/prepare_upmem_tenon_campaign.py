#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the canonical Tenon UPMEM uPIMulator campaign.

The exporter is deliberately hardware-free.  It asks the compiler-owned
physical plans for their runtime MRAM ABI and DPU translation unit, packs the
canonical inputs into that ABI, and emits exact heap-output oracles.  The
resulting ``programs/<workload>`` trees are directly consumable by the shared
uPIMulator ``run_case.py`` driver in the Tenon artifact bundle.

Host packing and CPU/DPU transfers are outside the archived kernel-cycle
metric.  This script records that qualification in every case rather than
hiding it in the physical fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from allo.pim.upmem_irregular import (
    GradientFormula,
    UPMEMFeatureGradientPlan,
    UPMEMHistogramPlan,
    UPMEMKMeansDistancesPlan,
    UPMEMSelectionFlagsPlan,
)
from allo.pim.upmem_physical import (
    UPMEMElementwisePlan,
    UPMEMGEMMPlan,
    UPMEMMatrixVectorPlan,
    UPMEMSumReductionPlan,
)
from allo.pim.upmem_physical_search import (
    DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
    UPMEMDataLayout,
    UPMEMOperandResidency,
    UPMEMOwnership,
    UPMEMPhysicalDecision,
    UPMEMPhysicalProblem,
    UPMEMPhysicalSearchResult,
    UPMEMPlanBuilderRejected,
    UPMEMPredicateLowering,
    rank_upmem_physical_candidates,
)


PREPARATION_VERSION = "tenon-upmem-campaign-v2"
ORACLE_SCOPE = "exact complete DPU output region for every launch"
WORKLOADS = (
    "va",
    "red",
    "mtv",
    "gemv",
    "geva",
    "ttv",
    "mmtv",
    "hist",
    "sel",
    "kmeans",
    "linear_reg",
    "logistic_reg",
    "1mm",
    "2mm",
    "3mm",
    "conv",
)
PHYSICAL_SEARCH_WORKLOADS = (
    "va",
    "mtv",
    "gemv",
    "geva",
    "ttv",
    "mmtv",
    "sel",
    "1mm",
    "2mm",
    "3mm",
    "conv",
)
ALLO_ROOT = Path(__file__).resolve().parents[1]
COMPILER_SOURCE_FILES = (
    "allo/__init__.py",
    "allo/pim/__init__.py",
    "allo/pim/upmem_calibration.py",
    "allo/pim/upmem_irregular.py",
    "allo/pim/upmem_physical.py",
    "allo/pim/upmem_physical_search.py",
    "allo/spmw_linear_layout.py",
    "allo/spmw_target.py",
    "scripts/prepare_upmem_tenon_campaign.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _unpack(data: bytes, kind: str) -> tuple[int, ...]:
    formats = {"i32": "i", "u32": "I", "i64": "q"}
    widths = {"i32": 4, "u32": 4, "i64": 8}
    if kind not in formats:
        raise ValueError(f"unsupported canonical dtype {kind!r}")
    if len(data) % widths[kind]:
        raise ValueError(f"{kind} byte stream has a partial element")
    return tuple(struct.unpack(f"<{len(data) // widths[kind]}{formats[kind]}", data))


def pack_i32(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return struct.pack(f"<{len(values)}i", *values)


def pack_u32(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return struct.pack(f"<{len(values)}I", *values)


def pack_i64(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return struct.pack(f"<{len(values)}q", *values)


def _signed_round_closest(numerator: int, denominator: int) -> int:
    """Match the archived one-iteration k-means host update rounding."""

    adjusted = (
        numerator - denominator // 2 if numerator < 0 else numerator + denominator // 2
    )
    magnitude = abs(adjusted) // denominator
    return -magnitude if adjusted < 0 else magnitude


class CanonicalInputs:
    """Typed, hash-checked access to the frozen canonical input bundle."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        manifest_path = self.root / "MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"canonical MANIFEST.json is missing: {manifest_path}"
            )
        self.manifest_bytes = manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes.decode("utf-8"))
        if not isinstance(self.manifest, dict):
            raise ValueError("canonical MANIFEST.json must contain an object")
        self.used: dict[str, dict[str, str]] = {}

    @property
    def manifest_sha256(self) -> str:
        return sha256_bytes(self.manifest_bytes)

    def read(
        self, workload: str, filename: str, kind: str, elements: int
    ) -> tuple[int, ...]:
        path = self.root / workload / filename
        data = path.read_bytes()
        values = _unpack(data, kind)
        if len(values) != elements:
            raise ValueError(
                f"{workload}/{filename}: expected {elements} elements, "
                f"found {len(values)}"
            )
        try:
            metadata = self.manifest[workload]["files"][filename]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"canonical manifest has no record for {workload}/{filename}"
            ) from error
        digest = sha256_bytes(data)
        expected = {
            "dtype": kind,
            "elements": elements,
            "bytes": len(data),
            "sha256": digest,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"canonical manifest {workload}/{filename} field {key!r} "
                    "does not match the file"
                )
        self.used.setdefault(workload, {})[filename] = digest
        return values

    def hashes_for(self, workload: str) -> dict[str, str]:
        return dict(sorted(self.used.get(workload, {}).items()))


def _region_dict(region: object) -> dict[str, Any]:
    if isinstance(region, dict):
        return dict(region)
    manifest = getattr(region, "manifest", None)
    if not callable(manifest):
        raise TypeError(f"unrecognized MRAM region {region!r}")
    value = manifest()
    if not isinstance(value, dict):
        raise TypeError("MRAM region manifest must be an object")
    return value


def _region(plan: object, name: str) -> dict[str, Any]:
    return _region_dict(getattr(plan, "mram_regions")[name])


def _write_region(
    image: bytearray, region: Mapping[str, Any], data: bytes, name: str
) -> None:
    offset, allocation = int(region["offset"]), int(region["bytes"])
    logical = int(region.get("logical_bytes", allocation))
    if len(data) > logical or logical > allocation:
        raise ValueError(
            f"{name}: {len(data)} payload bytes do not fit "
            f"the {logical}/{allocation}-byte region"
        )
    image[offset : offset + len(data)] = data


def _output_window(plan: object, expected_image: bytearray) -> tuple[int, bytes]:
    regions = sorted(
        (_region_dict(region) for region in getattr(plan, "output_regions").values()),
        key=lambda region: int(region["offset"]),
    )
    if not regions:
        raise ValueError("physical plan declares no output regions")
    for previous, current in zip(regions, regions[1:]):
        previous_end = int(previous["offset"]) + int(previous["bytes"])
        if previous_end != int(current["offset"]):
            raise ValueError("physical output regions must form one contiguous oracle")
    begin = int(regions[0]["offset"])
    end = int(regions[-1]["offset"]) + int(regions[-1]["bytes"])
    return begin, bytes(expected_image[begin:end])


def _expect(actual: Iterable[int], expected: Iterable[int], label: str) -> None:
    actual, expected = tuple(actual), tuple(expected)
    if actual != expected:
        raise ValueError(f"canonical {label} reference does not match its inputs")


def _matvec(
    matrix: tuple[int, ...], vector: tuple[int, ...], rows: int, columns: int
) -> tuple[int, ...]:
    return tuple(
        sum(
            matrix[row * columns + column] * vector[column] for column in range(columns)
        )
        for row in range(rows)
    )


def _matmul(
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
    rows: int,
    reduction: int,
    columns: int,
) -> tuple[int, ...]:
    return tuple(
        sum(
            lhs[row * reduction + k] * rhs[k * columns + column]
            for k in range(reduction)
        )
        for row in range(rows)
        for column in range(columns)
    )


def pack_fused_elementwise(
    plan: UPMEMElementwisePlan,
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
) -> bytes:
    """Pack ``[A_chunk, B_chunk]`` records with deterministic zero tails."""

    if len(lhs) != plan.elements or len(rhs) != plan.elements:
        raise ValueError("elementwise operands do not match the physical plan")
    values: list[int] = []
    chunk = plan.chunk_elements
    for block in range(plan.chunks):
        begin = block * chunk
        stop = min(begin + chunk, plan.elements)
        valid = stop - begin
        values.extend(lhs[begin:stop])
        values.extend([0] * (chunk - valid))
        values.extend(rhs[begin:stop])
        values.extend([0] * (chunk - valid))
    return pack_i32(values)


def pack_fused_matvec(
    plan: UPMEMMatrixVectorPlan,
    matrix: tuple[int, ...],
    vectors: tuple[int, ...],
) -> bytes:
    """Pack per-flat-row ``[A_chunk, correct_batch_x_chunk]`` records."""

    return pack_i32(plan.pack_fused_inputs(matrix, vectors))


def pack_separate_matvec(
    plan: UPMEMMatrixVectorPlan,
    matrix: tuple[int, ...],
    vectors: tuple[int, ...],
) -> tuple[bytes, bytes]:
    """Pack the exact contiguous ``A``/``B`` ABI for resident-vector MMTV."""

    if plan.vector_mode != "tasklet_private_wram":
        raise ValueError("separate MV packing requires tasklet_private_wram mode")
    if len(matrix) != plan.total_rows * plan.columns:
        raise ValueError("matrix operand does not match the resident MV plan")
    if len(vectors) != plan.batches * plan.columns:
        raise ValueError("vector operand does not match the resident MV plan")
    matrix_blob = pack_i32(matrix)
    vector_blob = pack_i32(vectors)
    for name, blob in (("matrix", matrix_blob), ("vector", vector_blob)):
        region = _region(plan, name)
        if region.get("storage") != "contiguous" or len(blob) != int(region["bytes"]):
            raise ValueError(
                f"{name}: resident MV payload does not exactly match its MRAM region"
            )
    return matrix_blob, vector_blob


def pack_fused_gemm(
    plan: UPMEMGEMMPlan,
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
) -> bytes:
    """Pack ``[A[Kc], B_col0[Kc], ..., B_colNc-1[Kc]]`` records."""

    if len(lhs) != plan.rows * plan.reduction:
        raise ValueError("GEMM lhs does not match the physical plan")
    if len(rhs) != plan.reduction * plan.columns:
        raise ValueError("GEMM rhs does not match the physical plan")
    values: list[int] = []
    for row in range(plan.padded_rows):
        for column_block in range(plan.column_tiles):
            for reduction_block in range(plan.reduction_tiles):
                reduction_begin = reduction_block * plan.reduction_tile
                for k in range(plan.reduction_tile):
                    valid = row < plan.rows and reduction_begin + k < plan.reduction
                    values.append(
                        lhs[row * plan.reduction + reduction_begin + k] if valid else 0
                    )
                for column_lane in range(plan.column_tile):
                    column = column_block * plan.column_tile + column_lane
                    for k in range(plan.reduction_tile):
                        reduction_index = reduction_begin + k
                        valid = (
                            row < plan.rows
                            and column < plan.columns
                            and reduction_index < plan.reduction
                        )
                        values.append(
                            rhs[reduction_index * plan.columns + column] if valid else 0
                        )
    return pack_i32(values)


def _new_images(plan: object) -> tuple[bytearray, bytearray]:
    image_bytes = int(getattr(plan, "mram_image_bytes"))
    return bytearray(image_bytes), bytearray(image_bytes)


def _elementwise_case(
    canonical: CanonicalInputs, workload: str, operation: str
) -> tuple[object, bytearray, bytearray, int]:
    lhs = canonical.read(workload, "a.i32.bin", "i32", 12_288)
    rhs = canonical.read(workload, "b.i32.bin", "i32", 12_288)
    expected = canonical.read(workload, "expected.i32.bin", "i32", 12_288)
    computed = (
        tuple(x + y for x, y in zip(lhs, rhs))
        if operation == "add"
        else tuple(2 * x - y for x, y in zip(lhs, rhs))
    )
    _expect(computed, expected, workload)
    plan = UPMEMElementwisePlan(12_288, operation=operation, interleaved=True)
    heap, output = _new_images(plan)
    _write_region(
        heap,
        _region(plan, "packed_operands"),
        pack_fused_elementwise(plan, lhs, rhs),
        "packed_operands",
    )
    if operation == "axpby" and "coefficients" in plan.mram_regions:
        _write_region(
            heap,
            _region(plan, "coefficients"),
            pack_i32((2, -1)),
            "coefficients",
        )
    padded = expected + (0,) * (plan.padded_elements - len(expected))
    _write_region(output, _region(plan, "output"), pack_i32(padded), "output")
    return plan, heap, output, 1


def _reduction_case(
    canonical: CanonicalInputs,
) -> tuple[object, bytearray, bytearray, int]:
    values = canonical.read("red", "a.i32.bin", "i32", 12_288)
    expected = canonical.read("red", "expected.i64.bin", "i64", 1)
    _expect((sum(values),), expected, "red")
    plan = UPMEMSumReductionPlan(12_288)
    heap, output = _new_images(plan)
    padded = values + (0,) * (plan.padded_elements - len(values))
    _write_region(heap, _region(plan, "input"), pack_i32(padded), "input")
    _write_region(output, _region(plan, "output"), pack_i64(expected), "output")
    return plan, heap, output, 1


def _matrix_vector_case(
    canonical: CanonicalInputs, workload: str
) -> tuple[object, bytearray, bytearray, int]:
    if workload in {"mtv", "gemv"}:
        batches, rows, columns = 1, 96, 128
        matrix = canonical.read(workload, "a.i32.bin", "i32", rows * columns)
        vectors = canonical.read(workload, "x.i32.bin", "i32", columns)
        expected = canonical.read(workload, "expected.i32.bin", "i32", rows)
        unscaled = _matvec(matrix, vectors, rows, columns)
        scale = 2 if workload == "gemv" else 1
        _expect(tuple(scale * value for value in unscaled), expected, workload)
    elif workload == "ttv":
        batches, rows, columns = 1, 192, 32
        matrix = canonical.read(workload, "a.i32.bin", "i32", rows * columns)
        vectors = canonical.read(workload, "x.i32.bin", "i32", columns)
        expected = canonical.read(workload, "expected.i32.bin", "i32", rows)
        _expect(_matvec(matrix, vectors, rows, columns), expected, workload)
        scale = 1
    elif workload == "mmtv":
        batches, rows, columns = 12, 16, 32
        matrix = canonical.read(workload, "a.i32.bin", "i32", batches * rows * columns)
        vectors = canonical.read(workload, "b.i32.bin", "i32", batches * columns)
        expected = canonical.read(workload, "expected.i32.bin", "i32", batches * rows)
        computed = tuple(
            sum(
                matrix[(batch * rows + row) * columns + column]
                * vectors[batch * columns + column]
                for column in range(columns)
            )
            for batch in range(batches)
            for row in range(rows)
        )
        _expect(computed, expected, workload)
        scale = 1
    else:  # pragma: no cover - internal dispatch is closed over WORKLOADS
        raise ValueError(workload)
    vector_mode = "tasklet_private_wram" if workload == "mmtv" else "fused_replicated"
    plan = UPMEMMatrixVectorPlan(
        rows=rows,
        columns=columns,
        batches=batches,
        scale_factor=scale,
        vector_mode=vector_mode,
    )
    heap, output = _new_images(plan)
    if plan.vector_mode == "tasklet_private_wram":
        matrix_blob, vector_blob = pack_separate_matvec(plan, matrix, vectors)
        _write_region(heap, _region(plan, "matrix"), matrix_blob, "matrix")
        _write_region(heap, _region(plan, "vector"), vector_blob, "vector")
    else:
        _write_region(
            heap,
            _region(plan, "packed_matrix_vector"),
            pack_fused_matvec(plan, matrix, vectors),
            "packed_matrix_vector",
        )
    _write_region(
        output,
        _region(plan, "output"),
        pack_i32(plan.pack_output(expected)),
        "output",
    )
    return plan, heap, output, 1


def _gemm_case(
    canonical: CanonicalInputs, workload: str
) -> tuple[object, bytearray, bytearray, int]:
    if workload == "conv":
        rows, reduction, columns, repeats = 192, 16, 32, 1
        lhs = canonical.read(workload, "patches.i32.bin", "i32", rows * reduction)
        rhs = canonical.read(workload, "filters.i32.bin", "i32", reduction * columns)
    else:
        rows, reduction, columns, repeats = 12, 64, 128, int(workload[0])
        lhs = canonical.read(workload, "a.i32.bin", "i32", rows * reduction)
        rhs = canonical.read(workload, "b.i32.bin", "i32", reduction * columns)
    expected = canonical.read(workload, "expected.i32.bin", "i32", rows * columns)
    _expect(_matmul(lhs, rhs, rows, reduction, columns), expected, workload)
    plan = UPMEMGEMMPlan(
        rows=rows,
        columns=columns,
        reduction=reduction,
        column_tile=16,
        reduction_tile=16,
    )
    heap, output = _new_images(plan)
    _write_region(
        heap,
        _region(plan, "packed_lhs_rhs"),
        pack_fused_gemm(plan, lhs, rhs),
        "packed_lhs_rhs",
    )
    physical = [0] * (plan.padded_rows * plan.padded_columns)
    for row in range(rows):
        physical[row * plan.padded_columns : row * plan.padded_columns + columns] = (
            expected[row * columns : (row + 1) * columns]
        )
    _write_region(output, _region(plan, "output"), pack_i32(physical), "output")
    return plan, heap, output, repeats


def _histogram_case(
    canonical: CanonicalInputs,
) -> tuple[object, bytearray, bytearray, int]:
    values = canonical.read("hist", "input.i32.bin", "i32", 12_288)
    expected = canonical.read("hist", "expected.u32.bin", "u32", 128)
    plan = UPMEMHistogramPlan(12_288, 128, 12)
    _expect(plan.reference_histogram(values), expected, "hist")
    heap, output = _new_images(plan)
    _write_region(heap, _region(plan, "input"), pack_i32(values), "input")
    _write_region(output, _region(plan, "histogram"), pack_u32(expected), "histogram")
    return plan, heap, output, 1


def _selection_case(
    canonical: CanonicalInputs,
) -> tuple[object, bytearray, bytearray, int]:
    values = canonical.read("sel", "input.i32.bin", "i32", 12_288)
    selected = canonical.read("sel", "expected.i32.bin", "i32", 6_224)
    expected_count = canonical.read("sel", "expected_count.u32.bin", "u32", 1)
    plan = UPMEMSelectionFlagsPlan(12_288)
    flags = plan.reference_flags(values)
    compacted = tuple(value for value in flags if value & 1)
    _expect(compacted, selected, "sel stable host compaction")
    _expect((len(compacted),), expected_count, "sel count")
    heap, output = _new_images(plan)
    _write_region(heap, _region(plan, "input"), pack_i32(values), "input")
    _write_region(output, _region(plan, "flags"), pack_i32(flags), "flags")
    return plan, heap, output, 1


def _kmeans_case(
    canonical: CanonicalInputs,
) -> tuple[object, bytearray, bytearray, int]:
    points = canonical.read("kmeans", "points.i32.bin", "i32", 120 * 8)
    initial = canonical.read("kmeans", "initial_centroids.i32.bin", "i32", 4 * 8)
    expected_centroids = canonical.read(
        "kmeans", "expected_centroids.i32.bin", "i32", 4 * 8
    )
    expected_counts = canonical.read("kmeans", "expected_counts.u32.bin", "u32", 4)
    plan = UPMEMKMeansDistancesPlan(120, 8, 4, max_abs_value=50)
    fused = plan.pack_fused_pairs(points, initial)
    distances = plan.reference_distances(fused)

    sums = [[0] * plan.dimension for _ in range(plan.clusters)]
    counts = [0] * plan.clusters
    for point in range(plan.points):
        chosen = min(
            range(plan.clusters),
            key=lambda cluster: distances[point * plan.clusters + cluster],
        )
        counts[chosen] += 1
        for feature in range(plan.dimension):
            sums[chosen][feature] += points[point * plan.dimension + feature]
    centroids = tuple(
        (
            0
            if counts[cluster] == 0
            else _signed_round_closest(sums[cluster][feature], counts[cluster])
        )
        for cluster in range(plan.clusters)
        for feature in range(plan.dimension)
    )
    _expect(centroids, expected_centroids, "kmeans host centroid update")
    _expect(tuple(counts), expected_counts, "kmeans host counts")
    heap, output = _new_images(plan)
    _write_region(heap, _region(plan, "fused_pairs"), pack_i32(fused), "fused_pairs")
    _write_region(output, _region(plan, "distances"), pack_i64(distances), "distances")
    return plan, heap, output, 1


def _gradient_case(
    canonical: CanonicalInputs, workload: str
) -> tuple[object, bytearray, bytearray, int]:
    samples = canonical.read(workload, "samples.i32.bin", "i32", 120 * 9)
    if workload == "linear_reg":
        formula = GradientFormula.LINEAR_FIXED_POINT
        expected = canonical.read(workload, "expected_gradient.i64.bin", "i64", 8)
    else:
        formula = GradientFormula.LOGISTIC_ZERO
        expected = canonical.read(
            workload,
            "expected_twice_gradient_at_zero.i64.bin",
            "i64",
            8,
        )
    plan = UPMEMFeatureGradientPlan(120, 8, formula, shift=5, overflow_shift=8)
    _expect(plan.reference_gradient(samples), expected, workload)
    heap, output = _new_images(plan)
    _write_region(heap, _region(plan, "samples"), pack_i32(samples), "samples")
    _write_region(output, _region(plan, "gradient"), pack_i64(expected), "gradient")
    return plan, heap, output, 1


def build_case(
    canonical: CanonicalInputs, workload: str
) -> tuple[object, bytes, int, bytes, int]:
    """Return plan, full input heap, output offset/blob, and launch count."""

    if workload == "va":
        plan, heap, output, repeats = _elementwise_case(canonical, workload, "add")
    elif workload == "geva":
        plan, heap, output, repeats = _elementwise_case(canonical, workload, "axpby")
    elif workload == "red":
        plan, heap, output, repeats = _reduction_case(canonical)
    elif workload in {"mtv", "gemv", "ttv", "mmtv"}:
        plan, heap, output, repeats = _matrix_vector_case(canonical, workload)
    elif workload in {"1mm", "2mm", "3mm", "conv"}:
        plan, heap, output, repeats = _gemm_case(canonical, workload)
    elif workload == "hist":
        plan, heap, output, repeats = _histogram_case(canonical)
    elif workload == "sel":
        plan, heap, output, repeats = _selection_case(canonical)
    elif workload == "kmeans":
        plan, heap, output, repeats = _kmeans_case(canonical)
    elif workload in {"linear_reg", "logistic_reg"}:
        plan, heap, output, repeats = _gradient_case(canonical, workload)
    else:
        raise ValueError(f"unknown canonical workload {workload!r}")
    output_offset, oracle = _output_window(plan, output)
    return plan, bytes(heap), output_offset, oracle, repeats


def _pointwise_search_decisions() -> tuple[UPMEMPhysicalDecision, ...]:
    return (
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 64, 16),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 128, 32),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 256, 64),
        UPMEMPhysicalDecision(UPMEMDataLayout.SEPARATE, 1024, 256),
        UPMEMPhysicalDecision(UPMEMDataLayout.FUSED_REPLICATED, 256, 32),
    )


def _matrix_vector_search_decisions() -> tuple[UPMEMPhysicalDecision, ...]:
    return (
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            64,
            16,
            vector_residency=UPMEMOperandResidency.SHARED_WRAM,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            128,
            16,
            vector_residency=UPMEMOperandResidency.FUSED_DMA_PACKET,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            64,
            16,
            vector_residency=UPMEMOperandResidency.TASKLET_PRIVATE_WRAM,
        ),
    )


def _gemm_search_decisions() -> tuple[UPMEMPhysicalDecision, ...]:
    fused = UPMEMOperandResidency.FUSED_DMA_PACKET
    return (
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            320,
            16,
            rhs_residency=fused,
            nc=4,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            576,
            16,
            rhs_residency=fused,
            nc=8,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1088,
            16,
            rhs_residency=fused,
            nc=16,
            kc=16,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.FUSED_REPLICATED,
            1056,
            8,
            rhs_residency=fused,
            nc=32,
            kc=8,
        ),
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            2048,
            64,
            rhs_residency=UPMEMOperandResidency.SHARED_WRAM,
            nc=128,
            kc=64,
        ),
    )


def _selection_search_decisions() -> tuple[UPMEMPhysicalDecision, ...]:
    return tuple(
        UPMEMPhysicalDecision(
            UPMEMDataLayout.SEPARATE,
            elements * 4,
            elements,
            predicate_lowering=lowering,
        )
        for elements, lowering in (
            (32, UPMEMPredicateLowering.TERNARY),
            (32, UPMEMPredicateLowering.BRANCHLESS_MASK),
            (32, UPMEMPredicateLowering.CONDITIONAL_ZERO),
            (16, UPMEMPredicateLowering.TERNARY),
            (64, UPMEMPredicateLowering.TERNARY),
            (128, UPMEMPredicateLowering.TERNARY),
            (8, UPMEMPredicateLowering.CONDITIONAL_ZERO),
            (16, UPMEMPredicateLowering.CONDITIONAL_ZERO),
            (64, UPMEMPredicateLowering.CONDITIONAL_ZERO),
            (16, UPMEMPredicateLowering.BRANCHLESS_MASK),
        )
    )


def _verify_decision_plan_binding(
    problem: UPMEMPhysicalProblem,
    decision: UPMEMPhysicalDecision,
    plan: object,
) -> None:
    """Fail closed unless every searched field agrees with emitted lowering."""

    if decision.tasklets != int(getattr(plan, "num_tasklets")):
        raise ValueError("physical search tasklet decision does not match plan")
    if decision.ownership is not UPMEMOwnership.CONTIGUOUS:
        raise ValueError("physical search selected unsupported ownership")
    if isinstance(plan, UPMEMElementwisePlan):
        expected_layout = (
            UPMEMDataLayout.FUSED_REPLICATED
            if plan.interleaved
            else UPMEMDataLayout.SEPARATE
        )
        expected_dma = plan.dma_bytes * (2 if plan.interleaved else 1)
        matches = (
            problem.kind.value == "pointwise"
            and problem.logical_elements == plan.elements
            and decision.layout is expected_layout
            and decision.dma_bytes == expected_dma
            and decision.chunk_elements == plan.chunk_elements
        )
    elif isinstance(plan, UPMEMMatrixVectorPlan):
        if plan.vector_mode == "fused_replicated":
            expected_layout = UPMEMDataLayout.FUSED_REPLICATED
            expected_residency = UPMEMOperandResidency.FUSED_DMA_PACKET
            expected_dma = plan.packet_bytes
        elif plan.vector_mode == "tasklet_private_wram":
            expected_layout = UPMEMDataLayout.SEPARATE
            expected_residency = UPMEMOperandResidency.TASKLET_PRIVATE_WRAM
            expected_dma = plan.dma_bytes
        else:  # pragma: no cover - plan validation closes this domain
            raise ValueError(f"unsupported MV vector mode {plan.vector_mode!r}")
        matches = (
            problem.kind.value == "matrix_vector"
            and problem.rows == plan.total_rows
            and problem.batches == plan.batches
            and problem.reduction == plan.columns
            and decision.layout is expected_layout
            and decision.vector_residency is expected_residency
            and decision.dma_bytes == expected_dma
            and decision.chunk_elements == plan.chunk_elements
        )
    elif isinstance(plan, UPMEMGEMMPlan):
        matches = (
            problem.kind.value == "gemm"
            and (problem.rows, problem.columns, problem.reduction)
            == (plan.rows, plan.columns, plan.reduction)
            and decision.layout is UPMEMDataLayout.FUSED_REPLICATED
            and decision.rhs_residency is UPMEMOperandResidency.FUSED_DMA_PACKET
            and decision.dma_bytes == plan.packet_bytes
            and decision.nc == plan.column_tile
            and decision.kc == plan.reduction_tile
            and decision.chunk_elements == plan.reduction_tile
        )
    elif isinstance(plan, UPMEMSelectionFlagsPlan):
        matches = (
            problem.kind.value == "selection_flags"
            and problem.logical_elements == plan.elements
            and decision.layout is UPMEMDataLayout.SEPARATE
            and decision.predicate_lowering is UPMEMPredicateLowering.CONDITIONAL_ZERO
            and decision.dma_bytes == 4 * plan.dma_elements
            and decision.chunk_elements == plan.dma_elements
        )
    else:
        raise TypeError(f"physical search cannot bind {type(plan).__name__}")
    if not matches:
        raise ValueError("physical search decision does not match emitted plan")


def build_physical_search(
    workload: str, emitted_plan: object
) -> tuple[UPMEMPhysicalSearchResult, tuple[UPMEMPhysicalDecision, ...]] | None:
    """Search modeled families and prove the winner is the emitted plan."""

    if workload not in PHYSICAL_SEARCH_WORKLOADS:
        return None
    if isinstance(emitted_plan, UPMEMElementwisePlan):
        operation_instructions = 1 if emitted_plan.operation == "add" else 2
        problem = UPMEMPhysicalProblem.pointwise(
            emitted_plan.elements,
            operation_instructions=operation_instructions,
        )
        decisions = _pointwise_search_decisions()

        def plan_builder(decision: UPMEMPhysicalDecision) -> object:
            interleaved = decision.layout is UPMEMDataLayout.FUSED_REPLICATED
            operand_dma = decision.dma_bytes // 2 if interleaved else decision.dma_bytes
            return UPMEMElementwisePlan(
                emitted_plan.elements,
                operation=emitted_plan.operation,
                dma_bytes=operand_dma,
                interleaved=interleaved,
                alpha=emitted_plan.alpha,
                beta=emitted_plan.beta,
                runtime_coefficients=emitted_plan.runtime_coefficients,
            )

    elif isinstance(emitted_plan, UPMEMMatrixVectorPlan):
        problem = UPMEMPhysicalProblem.matrix_vector(
            emitted_plan.total_rows,
            emitted_plan.columns,
            batches=emitted_plan.batches,
        )
        decisions = _matrix_vector_search_decisions()

        def plan_builder(decision: UPMEMPhysicalDecision) -> object:
            if (
                decision.layout is UPMEMDataLayout.FUSED_REPLICATED
                and decision.vector_residency is UPMEMOperandResidency.FUSED_DMA_PACKET
            ):
                vector_mode = "fused_replicated"
            elif (
                decision.layout is UPMEMDataLayout.SEPARATE
                and decision.vector_residency
                is UPMEMOperandResidency.TASKLET_PRIVATE_WRAM
            ):
                vector_mode = "tasklet_private_wram"
            else:
                raise UPMEMPlanBuilderRejected(
                    "MV lowering does not materialize the requested residency"
                )
            return UPMEMMatrixVectorPlan(
                rows=emitted_plan.rows,
                columns=emitted_plan.columns,
                batches=emitted_plan.batches,
                scale=emitted_plan.scale,
                scale_factor=emitted_plan.scale_factor,
                dma_bytes=decision.chunk_elements * 4,
                vector_mode=vector_mode,
            )

    elif isinstance(emitted_plan, UPMEMGEMMPlan):
        problem = UPMEMPhysicalProblem.gemm(
            emitted_plan.rows, emitted_plan.columns, emitted_plan.reduction
        )
        decisions = _gemm_search_decisions()

        def plan_builder(decision: UPMEMPhysicalDecision) -> object:
            if (
                decision.layout is not UPMEMDataLayout.FUSED_REPLICATED
                or decision.rhs_residency is not UPMEMOperandResidency.FUSED_DMA_PACKET
            ):
                raise UPMEMPlanBuilderRejected(
                    "current GEMM lowering materializes fused RHS packets only"
                )
            return UPMEMGEMMPlan(
                rows=emitted_plan.rows,
                columns=emitted_plan.columns,
                reduction=emitted_plan.reduction,
                column_tile=int(decision.nc),
                reduction_tile=int(decision.kc),
            )

    elif isinstance(emitted_plan, UPMEMSelectionFlagsPlan):
        problem = UPMEMPhysicalProblem.selection(emitted_plan.elements)
        decisions = _selection_search_decisions()

        def plan_builder(decision: UPMEMPhysicalDecision) -> object:
            if decision.predicate_lowering is not (
                UPMEMPredicateLowering.CONDITIONAL_ZERO
            ):
                raise UPMEMPlanBuilderRejected(
                    "selection-flags source materializes conditional-zero only"
                )
            return UPMEMSelectionFlagsPlan(
                emitted_plan.elements,
                num_tasklets=decision.tasklets,
                dma_elements=decision.chunk_elements,
            )

    else:  # pragma: no cover - guarded by the explicit workload set
        raise TypeError(f"no physical search for {type(emitted_plan).__name__}")

    result = rank_upmem_physical_candidates(
        problem,
        decisions,
        model=DEFAULT_UPMEM_PHYSICAL_COST_MODEL,
        plan_builder=plan_builder,
    )
    selected_plan = result.best.plan
    if selected_plan is None:
        raise ValueError("physical search winner did not materialize a plan")
    _verify_decision_plan_binding(problem, result.best.decision, selected_plan)
    if type(selected_plan) is not type(emitted_plan):
        raise ValueError("physical search winner has the wrong plan type")
    if selected_plan.manifest() != emitted_plan.manifest():
        raise ValueError("physical search winner manifest differs from emitted plan")
    if selected_plan.device_source() != emitted_plan.device_source():
        raise ValueError("physical search winner source differs from emitted plan")
    return result, decisions


def _git_output(arguments: list[str]) -> str:
    process = subprocess.run(
        ["git", "-C", str(ALLO_ROOT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout


def compiler_state() -> dict[str, object]:
    source_hashes = {
        relative: sha256_file(ALLO_ROOT / relative)
        for relative in COMPILER_SOURCE_FILES
    }
    status = _git_output(["status", "--short", "--", *COMPILER_SOURCE_FILES])
    return {
        "base_commit": _git_output(["rev-parse", "HEAD"]).strip(),
        "working_tree_modified": bool(status.strip()),
        "working_tree_status_sha256": sha256_bytes(status.encode("utf-8")),
        "source_sha256": source_hashes,
    }


def _case_notes(workload: str, repeats: int) -> dict[str, object]:
    notes: dict[str, object] = {
        "oracle_scope": ORACLE_SCOPE,
        "host_packing": (
            "canonical tensors are packed into the compiler-selected physical MRAM "
            "layout on the host"
        ),
        "timing_scope": (
            "uPIMulator DPU logic_cycle only; host packing, host computation, "
            "CPU-to-DPU transfers, and DPU-to-CPU transfers are excluded"
        ),
        "launches": repeats,
    }
    if workload == "sel":
        notes["partition_qualification"] = (
            "Tenon returns odd-value-or-zero flags and leaves stable compaction to "
            "the host, matching the archived ATiM DPU-kernel partition"
        )
    if workload == "kmeans":
        notes["partition_qualification"] = (
            "Tenon returns squared point/centroid distances and leaves argmin, "
            "counts, and rounded centroid update host-side, matching the archived "
            "ATiM DPU partition; the SimplePIM partition differs"
        )
    if workload == "mmtv":
        notes["row_layout_qualification"] = (
            "content-aware physical row permutation is superseded by the selected "
            "value-independent one-batch-per-tasklet schedule; canonical A and B "
            "remain in identity batch-major order"
        )
        notes["resident_vector_qualification"] = (
            "each tasklet loads its batch's 32-element vector once into private "
            "WRAM, streams sixteen matrix rows in 64-byte chunks, and performs one "
            "grouped 64-byte output write"
        )
    return notes


def _execution(output_offset: int) -> dict[str, object]:
    return {
        "dpus": [
            {
                "host_inputs": {},
                "host_outputs": {},
                "mram_heap_input": {"offset": 0, "file": "mram.input.bin"},
                "mram_heap_output": {
                    "offset": output_offset,
                    "file": "mram.output.expected.bin",
                },
            }
        ]
    }


def _write_physical_search_evidence(
    source: Path,
    workload: str,
    plan: object,
    task_path: Path,
    plan_path: Path,
) -> dict[str, object]:
    search = build_physical_search(workload, plan)
    if search is None:
        return {
            "status": "not_applicable",
            "reason": (
                f"{type(plan).__name__} is outside the calibrated pointwise, "
                "matrix-vector, GEMM, and selection-flags search families"
            ),
        }

    result, decisions = search
    model_path = source / "physical-cost-model.json"
    write_json(model_path, result.model.manifest())
    model_sha256 = sha256_file(model_path)
    plan_sha256 = sha256_file(plan_path)
    task_sha256 = sha256_file(task_path)
    search_document = result.manifest()
    search_document.update(
        {
            "workload": workload,
            "preparation_version": PREPARATION_VERSION,
            "decision_space": {
                "schema": "tenon-upmem-ordered-physical-decision-space-v1",
                "order_is_fingerprint_input": True,
                "assignments": [
                    {
                        "ordinal": ordinal,
                        "decision_fingerprint": decision.fingerprint,
                        "decision": decision.manifest(),
                    }
                    for ordinal, decision in enumerate(decisions)
                ],
            },
            "selected": {
                "candidate_fingerprint": result.best.fingerprint,
                "decision_fingerprint": result.best.decision.fingerprint,
                "decision": result.best.decision.manifest(),
                "predicted_logic_cycles": result.best.predicted_logic_cycles,
                "calibration_status": result.best.estimate.calibration_status,
                "calibration_evidence_ids": list(
                    result.best.estimate.calibration_evidence_ids
                ),
            },
            "cost_model_binding": {
                "file": "physical-cost-model.json",
                "sha256": model_sha256,
                "model_fingerprint": result.model.fingerprint,
            },
            "emitted_plan_binding": {
                "verification": (
                    "selected materialized plan manifest and device source exactly "
                    "equal the emitted compiler plan before campaign decoration"
                ),
                "physical_plan_file": "physical-plan.json",
                "physical_plan_sha256": plan_sha256,
                "device_source_file": "task.c",
                "device_source_sha256": task_sha256,
            },
            "calibration_qualification": (
                "multi-batch matrix-vector evidence is exact-shape-bound by batch "
                "count, flattened rows, and reduction size; established single-batch "
                "MTV/GEMV strata remain generalized, and simulator measurements "
                "remain authoritative for performance claims"
            ),
        }
    )
    search_path = source / "physical-search.json"
    write_json(search_path, search_document)
    return {
        "status": "selected_and_bound",
        "search_file": "source/physical-search.json",
        "search_sha256": sha256_file(search_path),
        "search_fingerprint": result.search_fingerprint,
        "selected_candidate_fingerprint": result.best.fingerprint,
        "selected_decision_fingerprint": result.best.decision.fingerprint,
        "cost_model_file": "source/physical-cost-model.json",
        "cost_model_sha256": model_sha256,
        "model_fingerprint": result.model.fingerprint,
    }


def _write_mmtv_row_layout_evidence(
    source: Path,
    fixture: Path,
    canonical: CanonicalInputs,
    workload: str,
    plan: object,
    task_path: Path,
    plan_path: Path,
) -> dict[str, object]:
    if workload != "mmtv":
        return {
            "status": "not_applicable",
            "reason": "content-aware row-layout search is specific to canonical MMTV",
        }
    if not isinstance(plan, UPMEMMatrixVectorPlan):
        raise TypeError("MMTV row-layout search requires UPMEMMatrixVectorPlan")
    if plan.vector_mode != "tasklet_private_wram":
        raise ValueError("canonical MMTV must emit tasklet_private_wram mode")
    if not plan.is_identity_row_layout:
        raise ValueError("resident MMTV must retain value-independent identity rows")
    if (plan.batches, plan.rows, plan.columns, plan.dma_bytes) != (12, 16, 32, 64):
        raise ValueError("resident MMTV plan shape/DMA differs from calibrated shape")

    matrix = canonical.read("mmtv", "a.i32.bin", "i32", 12 * 16 * 32)
    vectors = canonical.read("mmtv", "b.i32.bin", "i32", 12 * 32)
    expected = canonical.read("mmtv", "expected.i32.bin", "i32", 12 * 16)
    matrix_blob, vector_blob = pack_separate_matvec(plan, matrix, vectors)
    fixture_input_path = fixture / "mram.input.bin"
    fixture_input = fixture_input_path.read_bytes()
    for name, blob in (("matrix", matrix_blob), ("vector", vector_blob)):
        region = _region(plan, name)
        offset = int(region["offset"])
        if fixture_input[offset : offset + len(blob)] != blob:
            raise ValueError(f"resident MMTV {name} differs from fixture packing")
    packed_output = pack_i32(plan.pack_output(expected))
    fixture_output_path = fixture / "mram.output.expected.bin"
    if fixture_output_path.read_bytes() != packed_output:
        raise ValueError("resident MMTV output oracle differs from identity row order")

    path = source / "row-layout-search.json"
    if path.exists():
        raise ValueError("resident MMTV must not emit row-layout-search.json")
    return {
        "status": "not_applicable",
        "reason": (
            "superseded by the value-independent tasklet-private-WRAM physical "
            "schedule selected for canonical MMTV"
        ),
        "superseded_search": "content-aware physical-row permutation",
        "selected_vector_mode": plan.vector_mode,
        "identity_row_layout": True,
        "physical_plan_sha256": sha256_file(plan_path),
        "device_source_sha256": sha256_file(task_path),
        "fixture_input_sha256": sha256_file(fixture_input_path),
        "fixture_output_sha256": sha256_file(fixture_output_path),
        "matrix_sha256": canonical.hashes_for("mmtv")["a.i32.bin"],
        "vectors_sha256": canonical.hashes_for("mmtv")["b.i32.bin"],
    }


def prepare_campaign(input_root: Path, output_root: Path) -> tuple[Path, ...]:
    """Create a complete deterministic 16-case ``programs`` tree."""

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    programs = output_root / "programs"
    programs.mkdir()
    canonical = CanonicalInputs(input_root)
    state = compiler_state()
    cases: list[Path] = []

    for workload in WORKLOADS:
        plan, heap_input, output_offset, oracle, repeats = build_case(
            canonical, workload
        )
        case = programs / workload
        source = case / "source"
        fixture = case / "fixture"
        source.mkdir(parents=True)
        fixture.mkdir()

        task = source / "task.c"
        task.write_text(plan.device_source(), encoding="utf-8")
        plan_document = dict(plan.manifest())
        plan_document.update(
            {
                "schema": "tenon-upmem-physical-plan-v1",
                "workload": workload,
                "preparation_version": PREPARATION_VERSION,
                "sequential_executions": repeats,
            }
        )
        plan_path = source / "physical-plan.json"
        write_json(plan_path, plan_document)
        physical_search = _write_physical_search_evidence(
            source, workload, plan, task, plan_path
        )

        (fixture / "mram.input.bin").write_bytes(heap_input)
        (fixture / "mram.output.expected.bin").write_bytes(oracle)
        manifest = {
            "schema_version": 1,
            "num_dpus": 1,
            "num_tasklets": 12,
            "executions": [
                _execution(output_offset) for _execution_index in range(repeats)
            ],
        }
        manifest_path = fixture / "manifest.json"
        write_json(manifest_path, manifest)
        row_layout_search = _write_mmtv_row_layout_evidence(
            source,
            fixture,
            canonical,
            workload,
            plan,
            task,
            plan_path,
        )

        source_hashes = hash_tree(source)
        fixture_hashes = hash_tree(fixture)
        provenance = {
            "compiler": "Tenon",
            "compiler_commit": state["base_commit"],
            "compiler_state": state,
            "workload": workload,
            "preparation_version": PREPARATION_VERSION,
            "fixture_manifest": "fixture/manifest.json",
            "fixture_manifest_sha256": sha256_file(manifest_path),
            "canonical_input_manifest_sha256": canonical.manifest_sha256,
            "canonical_input_sha256": canonical.hashes_for(workload),
            "dpu_source_sha256": sha256_file(task),
            "physical_plan_sha256": sha256_file(plan_path),
            "physical_search": physical_search,
            "row_layout_search": row_layout_search,
            "source_sha256": source_hashes,
            "fixture_sha256": fixture_hashes,
            "num_dpus": 1,
            "num_tasklets": 12,
            "num_executions": repeats,
            "mram_heap_input_bytes": len(heap_input),
            "mram_heap_output_offset": output_offset,
            "mram_heap_output_bytes": len(oracle),
            "oracle_scope": ORACLE_SCOPE,
            "notes": _case_notes(workload, repeats),
        }
        write_json(case / "provenance.json", provenance)
        cases.append(case)
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    cases = prepare_campaign(args.input_root, args.output_root)
    print(f"prepared {len(cases)} Tenon UPMEM cases under {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
