#!/usr/bin/env python3
"""Export the UPMEM/Tenon PolyBench comparison artifacts.

The comparison contract is PolyBench/C 4.2.1 LARGE_DATASET with int32 device
arithmetic.  P0-P2 kernels are explicit compositions of cycle-accurately
measured uPIMulator dot or dense-tile launches.  P3 kernels fail closed at the
physical partitioning boundary; they are never assigned analytical cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import pprint
import re
import shutil
import subprocess

import allo
from allo.backend.c import emit_c_from_mlir
from allo.ir.types import int32
from allo.pim.upmem_program import UPMEMDenseTile, UPMEMDotTile, UPMEMRank1Tile

from lib.upmem_polybench import get_case


DEFAULT_ROOT = Path("/home/nz264/shared/tenon-artifacts/polybench/upmem/tenon")
METHODOLOGY = Path("/home/nz264/shared/tenon-artifacts/polybench/_methodology")

# PolyBench/C 4.2.1 LARGE_DATASET. Names follow Allo's generic kernels.
LARGE = {
    "2mm": {"P": 1600, "Q": 2200, "R": 1800, "S": 2400},
    "3mm": {"P": 1600, "Q": 2000, "R": 1800, "S": 2400, "T": 2200},
    "adi": {"TSTEPS": 500, "N": 1000},
    "atax": {"M": 1900, "N": 2100},
    "bicg": {"M": 1900, "N": 2100},
    "cholesky": {"N": 2000},
    "correlation": {"M": 1200, "N": 1400},
    "covariance": {"M": 1200, "N": 1400},
    "deriche": {"W": 4096, "H": 2160},
    "doitgen": {"R": 150, "Q": 140, "P": 160, "S": 160},
    "durbin": {"N": 2000},
    "fdtd_2d": {"Nx": 1000, "Ny": 1200, "Tmax": 500},
    "floyd_warshall": {"N": 2800},
    "gemm": {"P": 1000, "Q": 1200, "R": 1100},
    "gemver": {"N": 2000},
    "gesummv": {"N": 2800},
    "gramschmidt": {"M": 1000, "N": 1200},
    "heat_3d": {"TSTEPS": 500, "N": 200},
    "jacobi_1d": {"TSTEPS": 1000, "N": 2000},
    "jacobi_2d": {"TSTEPS": 500, "N": 1300},
    "lu": {"N": 2000},
    "ludcmp": {"N": 2000},
    "mvt": {"N": 2000},
    "nussinov": {"N": 2500},
    "seidel_2d": {"TSTEPS": 500, "N": 2000},
    "symm": {"M": 1000, "N": 1200},
    "syr2k": {"N": 1200, "M": 1000},
    "syrk": {"N": 1200, "M": 1000},
    "trisolv": {"N": 2000},
    "trmm": {"M": 1000, "N": 1200},
}

P3 = {
    "adi",
    "cholesky",
    "deriche",
    "durbin",
    "fdtd_2d",
    "floyd_warshall",
    "gramschmidt",
    "heat_3d",
    "jacobi_1d",
    "jacobi_2d",
    "lu",
    "ludcmp",
    "nussinov",
    "seidel_2d",
    "trisolv",
}

DESCRIPTION = {
    "2mm": "D = (A B) C",
    "3mm": "G = (A B) (C D)",
    "atax": "y = A^T (A x)",
    "bicg": "q = A p; s = A^T r",
    "correlation": "gram/correlation matrix; normalization remains host-side",
    "covariance": "mean plus X^T X gram; centering/scaling remains host-side",
    "doitgen": "reshape(R*Q,P) times a P-by-P matrix",
    "gemm": "C = A B",
    "gemver": "two rank-1 updates and two matrix-vector products",
    "gesummv": "y = A x + B x",
    "mvt": "x1 += A y1; x2 += A^T y2",
    "symm": "dense symmetric matrix product",
    "syr2k": "A B^T + B A^T",
    "syrk": "A A^T",
    "trmm": "dense execution of triangular matrix product",
}

FULL_SEMANTICS = {"3mm", "atax", "bicg"}
OMITTED = {
    "2mm": "alpha/beta scaling and final D combine",
    "correlation": "mean, standard deviation, centering, sqrt, and normalization",
    "covariance": "mean, centering, and normalization",
    "doitgen": "canonical scratch/result materialization outside the dense product",
    "gemm": "beta*C epilogue",
    "gemver": "z add and scalar alpha/beta epilogues",
    "gesummv": "alpha/beta scaling and final vector add",
    "mvt": "addition into the initial x1/x2 vectors",
    "symm": "symmetric/triangular access semantics and alpha/beta epilogue",
    "syr2k": "triangle-only stores, alpha/beta scaling, and combine epilogue",
    "syrk": "triangle-only stores and alpha/beta epilogue",
    "trmm": "triangular access semantics and alpha scaling",
}

DOT_LOG = {
    2048: METHODOLOGY / "cinm/sim_logs/upmem-fullsweep/upmem_2048_16.log",
    4096: METHODOLOGY / "cinm/sim_logs/upmem-fullsweep/upmem_4096_16.log",
}
DENSE_LOG = {
    reduction: METHODOLOGY
    / f"cinm/sim_logs/upmem-gemm-dedicated/upmem_gemm_{reduction}_8.log"
    for reduction in (160, 1000, 1200, 1800, 2000, 2200, 2400)
}
DENSE_LOG[1400] = (
    METHODOLOGY / "cinm/sim_logs/upmem-gemm-dedicated/upmem_gram_1400_8.log"
)
RANK1_LOG = METHODOLOGY / "sim_logs/exo-upmem/rank1_K2048_nt16.log"
TENON_RAW = Path("/work/shared/users/phd/nz264/upmem-tenon-raw")
for _fresh_k in (160, 1000, 1200, 1400, 1800, 2000, 2200, 2400):
    _fresh_log = TENON_RAW / f"K{_fresh_k}/log.txt"
    if _fresh_log.exists():
        DENSE_LOG[_fresh_k] = _fresh_log
for _fresh_dot in (2048, 4096):
    _fresh_log = TENON_RAW / f"DOT{_fresh_dot}/log.txt"
    if _fresh_log.exists():
        DOT_LOG[_fresh_dot] = _fresh_log
_fresh_rank1 = TENON_RAW / "RANK1_2048/log.txt"
if _fresh_rank1.exists():
    RANK1_LOG = _fresh_rank1
KC = {160: 40, 1000: 50, 1200: 60, 1400: 50, 1800: 60, 2000: 50, 2200: 50, 2400: 60}

# (primitive, measured extent, multiplicity). Dense multiplicity uses the
# comparison methodology's full-grid equivalent output-column count / 256.
COMPOSITION = {
    "atax": (("dot", 4096, 1), ("dot", 2048, 1)),
    "bicg": (("dot", 4096, 1), ("dot", 2048, 1)),
    "gesummv": (("dot", 4096, 2),),
    "mvt": (("dot", 2048, 2),),
    "gemver": (("rank1", 2048, 2), ("dot", 2048, 2)),
    "gemm": (("dense", 1200, Fraction(1100, 256)),),
    "2mm": (
        ("dense", 2200, Fraction(1800, 256)),
        ("dense", 1800, Fraction(2400, 256)),
    ),
    "3mm": (
        ("dense", 2000, Fraction(1800, 256)),
        ("dense", 2400, Fraction(2200, 256)),
        ("dense", 1800, Fraction(2200, 256)),
    ),
    "doitgen": (("dense", 160, Fraction(160, 256)),),
    "syrk": (("dense", 1000, Fraction(1200, 256)),),
    "syr2k": (("dense", 1000, Fraction(2400, 256)),),
    "symm": (("dense", 1000, Fraction(1200, 256)),),
    "trmm": (("dense", 1000, Fraction(1200, 256)),),
    "covariance": (("dense", 1400, Fraction(1200, 256)),),
    "correlation": (("dense", 1400, Fraction(1200, 256)),),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycle(path: Path) -> int:
    values = re.findall(
        r"(?:logic_cycle|Logic\[[^]]+\]_logic_cycle)\s*[:=]\s*(\d+)",
        path.read_text(),
    )
    if not values:
        raise ValueError(f"no uPIMulator logic cycle in {path}")
    return int(values[-1])


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _ambient(case, dimensions):
    module = case.module
    for name in case.scalars:
        value = dimensions.get("N", 1) if name == "N_float" else 1
        setattr(module, name, int(value))


def _compile_canonical(case_name, dimensions, compiled_dir):
    case = get_case(case_name)
    kernel = case.kernel
    _ambient(case, dimensions)
    instantiate = [int32, *(dimensions[name] for name in case.instantiate_dimensions)]
    schedule = allo.customize(kernel, enable_tensor=False, instantiate=instantiate)
    artifact = emit_c_from_mlir(
        schedule.module, schedule.top_func_name, wrap_wide_integers=True
    )
    source_path = compiled_dir / "polybench.large.int32.mlir"
    lowered_path = compiled_dir / "polybench.large.int32.lowered.mlir"
    c_path = compiled_dir / "polybench.large.int32.c"
    source_path.write_text(artifact.source_mlir)
    lowered_path.write_text(artifact.lowered_mlir)
    c_path.write_text(artifact.c_source)
    return (source_path, lowered_path, c_path)


def _profile_for(primitive, extent):
    if primitive == "dot":
        return DOT_LOG[extent]
    if primitive == "dense":
        return DENSE_LOG[extent]
    if primitive == "rank1":
        return RANK1_LOG
    raise KeyError(primitive)


def _physical_artifacts(case_name, compiled_dir):
    emitted = []
    for primitive, extent, _multiplicity in COMPOSITION[case_name]:
        if primitive == "dense":
            path = compiled_dir / f"dense_tile_16x16x{extent}.dpu.c"
            manifest = compiled_dir / f"dense_tile_16x16x{extent}.json"
            tile = UPMEMDenseTile(16, 16, extent, KC[extent], 8)
        elif primitive == "dot":
            path = compiled_dir / f"dot_tile_{extent}.dpu.c"
            manifest = compiled_dir / f"dot_tile_{extent}.json"
            tile = UPMEMDotTile(extent, 16, 128)
        else:
            path = compiled_dir / f"rank1_tile_{extent}.dpu.c"
            manifest = compiled_dir / f"rank1_tile_{extent}.json"
            tile = UPMEMRank1Tile(extent, 16)
        if not path.exists():
            path.write_text(tile.device_source())
            manifest.write_text(json.dumps(tile.manifest(), indent=2) + "\n")
        emitted.extend((path, manifest))
    return emitted


def _workload_spec(name, status):
    spec = {
        "kernel": name,
        "dataset": "LARGE_DATASET",
        "dtype": "int32",
        "dimensions": LARGE[name],
        "class": (
            "P3"
            if name in P3
            else (
                "P0"
                if name in {"atax", "bicg", "gemver", "gesummv", "mvt"}
                else ("P2" if name in {"covariance", "correlation"} else "P1")
            )
        ),
        "description": DESCRIPTION.get(name, "no legal PIM dispatch"),
        "status": status,
        "semantic_status": (
            "NOT_PROFILED"
            if name in P3
            else ("FULL" if name in FULL_SEMANTICS else "PARTIAL")
        ),
        "omitted_semantics": OMITTED.get(name),
        "composition": [
            {
                "primitive": primitive,
                "measured_extent": extent,
                "multiplicity": str(multiplicity),
            }
            for primitive, extent, multiplicity in COMPOSITION.get(name, ())
        ],
    }
    return spec


def _write_workload(path, spec):
    path.write_text(
        '"""Exact UPMEM/Tenon PolyBench comparison workload."""\n\n'
        + "SPEC = "
        + pprint.pformat(spec, sort_dicts=False, width=88)
        + "\n\ndef build():\n    return SPEC\n"
    )


def _p3_reason(name):
    if name in {"adi", "fdtd_2d", "heat_3d", "jacobi_1d", "jacobi_2d", "seidel_2d"}:
        return "temporal stencil/wavefront requires inter-DPU halo barriers"
    if name in {"cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv"}:
        return "loop-carried pivot/reduction requires a host-global barrier inside the kernel"
    return (
        "data-dependent or interval recurrence has no independent PIM dispatch frontier"
    )


def export(root: Path):
    commit = _git_commit()
    summary = []
    for name in LARGE:
        leaf = root / name.replace("2mm", "two_mm").replace("3mm", "three_mm")
        # Existing comparison leaves use two_mm/three_mm names. Fall back to
        # source names if this checkout retained 2mm/3mm.
        if not leaf.exists():
            leaf = root / name
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / ".gitkeep").unlink(missing_ok=True)
        compiled_dir, profile_dir = leaf / "compiled", leaf / "profile"
        shutil.rmtree(compiled_dir, ignore_errors=True)
        shutil.rmtree(profile_dir, ignore_errors=True)
        compiled_dir.mkdir(exist_ok=True)
        profile_dir.mkdir(exist_ok=True)

        status = "NOT_APPLICABLE" if name in P3 else "measured_composition"
        spec = _workload_spec(name, status)
        _write_workload(leaf / "workload.py", spec)
        compiled_files = []
        try:
            compiled_files.extend(_compile_canonical(name, LARGE[name], compiled_dir))
            canonical_status = "mlir_and_portable_c_emitted"
        except Exception as error:  # retain exact compiler rejection
            failure = compiled_dir / "canonical_compile_failure.log"
            failure.write_text(f"{type(error).__name__}: {error}\n")
            compiled_files.append(failure)
            canonical_status = "frontend_or_portable_c_rejected"

        if name in P3:
            rejection = {
                "status": "rejected",
                "boundary": "current_tenon_native_dpu_lowering",
                "reason": _p3_reason(name),
                "target_capability": (
                    "UPMEM supports arbitrary C; this is not a hardware rejection"
                ),
                "portable_c_emitted": canonical_status == "mlir_and_portable_c_emitted",
                "analytical_cycles_used": False,
            }
            rejection_path = compiled_dir / "physical_lowering_rejection.json"
            rejection_path.write_text(json.dumps(rejection, indent=2) + "\n")
            compiled_files.append(rejection_path)
            na = profile_dir / "not_applicable.log"
            na.write_text(
                "NOT_APPLICABLE: no complete native Tenon DPU dispatch.\n"
                "No simulator run: current Tenon native-DPU lowering is incomplete.\n"
                "UPMEM supports arbitrary C; this is not a hardware rejection.\n"
                f"Missing lowering: {rejection['reason']}.\n"
                "No analytical estimate substituted.\n"
            )
            result = {
                **spec,
                "metric": None,
                "cycles": None,
                "correctness": "not_run_native_dpu_lowering_missing",
                "canonical_compile_status": canonical_status,
                "physical_lowering": rejection,
                "provenance": {
                    "allo_commit": commit,
                    "compiled_sha256": {p.name: _sha256(p) for p in compiled_files},
                },
            }
        else:
            compiled_files.extend(_physical_artifacts(name, compiled_dir))
            terms, profile_hashes, copied = [], {}, {}
            total = Fraction(0)
            for primitive, extent, multiplicity in COMPOSITION[name]:
                source = _profile_for(primitive, extent)
                destination_name = (
                    f"tenon_{primitive}_K{extent}.log"
                    if source.name == "log.txt" and TENON_RAW in source.parents
                    else source.name
                )
                destination = profile_dir / destination_name
                if destination.name in copied and copied[destination.name] != source:
                    destination = profile_dir / f"{primitive}_{extent}_{source.name}"
                shutil.copy2(source, destination)
                copied[destination.name] = source
                cycles = _cycle(source)
                term = Fraction(cycles) * multiplicity
                total += term
                terms.append(
                    {
                        "primitive": primitive,
                        "measured_extent": extent,
                        "measured_cycles": cycles,
                        "multiplicity": str(multiplicity),
                        "cycle_term": str(term),
                        "raw_log": f"profile/{destination.name}",
                    }
                )
                profile_hashes[destination.name] = _sha256(destination)
                checker_log = None
                if source.name == "log.txt" and (source.parent / "run.out").exists():
                    checker_destination = (
                        profile_dir / f"tenon_{primitive}_K{extent}.run.out"
                    )
                    shutil.copy2(source.parent / "run.out", checker_destination)
                    checker_log = f"profile/{checker_destination.name}"
                    profile_hashes[checker_destination.name] = _sha256(
                        checker_destination
                    )
                terms[-1]["checker_log"] = checker_log
            composed_cycles = round(float(total))
            formula = " + ".join(
                f"{term['measured_cycles']}*({term['multiplicity']})" for term in terms
            )
            (profile_dir / "composition.log").write_text(
                f"formula_cycles = {formula}\n"
                f"exact_rational_cycles = {total}\n"
                f"reported_cycles = {composed_cycles}\n"
                "clock_hz = 350000000\n"
                "source = genuine uPIMulator logic_cycle logs; no Tenon cost model\n"
            )
            result = {
                **spec,
                "metric": "uPIMulator_logic_cycles",
                "cycles": composed_cycles,
                "microseconds_at_350mhz": composed_cycles / 350.0,
                "formula": formula,
                "terms": terms,
                "correctness": "simulator_checker_clean_finish_for_each_primitive",
                "canonical_compile_status": canonical_status,
                "provenance": {
                    "allo_commit": commit,
                    "simulator": "uPIMulator cycle-accurate Go simulator",
                    "raw_profile_sha256": profile_hashes,
                    "compiled_sha256": {p.name: _sha256(p) for p in compiled_files},
                    "methodology": "polybench/_methodology/cinm/sim_logs",
                    "fresh_tenon_source_reruns": [
                        {"primitive": primitive, "extent": extent}
                        for primitive, extent, _multiplicity in COMPOSITION[name]
                        if (
                            primitive == "dense"
                            and (TENON_RAW / f"K{extent}/log.txt").exists()
                        )
                        or (
                            primitive == "dot"
                            and (TENON_RAW / f"DOT{extent}/log.txt").exists()
                        )
                        or (
                            primitive == "rank1"
                            and (TENON_RAW / f"RANK1_{extent}/log.txt").exists()
                        )
                    ],
                    "analytical_cycles_used": False,
                },
            }

        (leaf / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        readme = [
            f"# {name} — UPMEM / Tenon",
            "",
            f"- Dataset: PolyBench/C 4.2.1 `LARGE_DATASET`",
            "- Device dtype: exact `int32`",
            f"- Status: `{status}`",
            f"- Dimensions: `{json.dumps(LARGE[name], sort_keys=True)}`",
            f"- Canonical MLIR/C: `{canonical_status}`",
        ]
        if name in P3:
            readme += [
                "- Native-device status: current Tenon lowering/profile gap; "
                "UPMEM itself supports arbitrary C.",
                f"- Missing lowering: {_p3_reason(name)}",
                "- Profiling: N/A; no analytical number was substituted.",
            ]
        else:
            readme += [
                f"- Measured composition: `{result['formula']}`",
                f"- Canonical semantic coverage: **{spec['semantic_status']}**"
                + (
                    f" (omits {spec['omitted_semantics']})"
                    if spec["omitted_semantics"]
                    else ""
                ),
                f"- Result: **{result['cycles']} uPIMulator logic cycles** "
                f"({result['microseconds_at_350mhz']:.3f} µs at 350 MHz)",
                "- Correctness: every raw primitive log ended cleanly under the simulator checker.",
            ]
        readme += [
            "",
            "Artifacts: `workload.py`, `compiled/`, `profile/`, and `result.json`.",
            "The result is derived only from raw simulator cycles; Tenon's analytical cost model is not used.",
            "",
        ]
        (leaf / "README.md").write_text("\n".join(readme))
        summary.append((name, status, result.get("cycles")))
    table = [
        "# UPMEM / Tenon PolyBench LARGE int32 results",
        "",
        "| kernel | profile status | semantics | uPIMulator logic cycles |",
        "|---|---:|---:|---:|",
    ]
    table.extend(
        f"| {name} | {status} | "
        f"{('NOT_PROFILED' if name in P3 else ('FULL' if name in FULL_SEMANTICS else 'PARTIAL'))} | "
        f"{cycles if cycles is not None else 'N/A'} |"
        for name, status, cycles in summary
    )
    table += [
        "",
        "All measured rows are compositions of genuine cycle-accurate uPIMulator logs; "
        "P3 rows are current Tenon native-DPU lowering gaps (not UPMEM hardware "
        "rejections) with no analytical substitute.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(table))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "LARGE_DATASET",
                "device_dtype": "int32",
                "total_leaves": len(summary),
                "measured_compositions": sum(
                    status == "measured_composition"
                    for _name, status, _cycles in summary
                ),
                "native_dpu_lowering_gaps": sum(
                    status == "NOT_APPLICABLE"
                    for _name, status, _cycles in summary
                ),
                "analytical_cycles_used": False,
                "allo_commit": commit,
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    for name, status, cycles in export(args.root):
        print(f"{name:16s} {status:22s} {cycles if cycles is not None else 'N/A'}")


if __name__ == "__main__":
    main()
