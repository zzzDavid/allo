# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gemini-II payload carries only what a public runtime builds (G2-11)."""

from pathlib import Path
import re

import allo.pim as pim
from allo.pim import apu_g2_pipeline, apu_g2_recipe
from allo.pim.costs import apu_g2 as apu_g2_costs

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "allo" / "pim" / "templates" / "apu_g2"

# Host executable -> (host source, ARC task source, runtime that launches it).
RUNTIMES = {
    "tenon_apu_g2_u16_add": ("host.cc", "apu_g2_u16_add.cc", "apu_g2_runtime.py"),
    "tenon_apu_g2_typed": (
        "host_typed.cc",
        "apu_g2_typed.cc",
        "apu_g2_typed_runtime.py",
    ),
    "tenon_apu_g2_composed_dot": (
        "host_composed_dot.cc",
        "apu_g2_composed_dot.cc",
        "apu_g2_composed_runtime.py",
    ),
}


def test_template_has_one_host_and_one_task_per_public_runtime():
    hosts = sorted(path.name for path in TEMPLATE.glob("*.cc"))
    tasks = sorted(path.name for path in (TEMPLATE / "device").glob("*.cc"))
    assert hosts == sorted(item[0] for item in RUNTIMES.values())
    assert tasks == sorted(item[1] for item in RUNTIMES.values())
    host_cmake = (TEMPLATE / "CMakeLists.txt").read_text()
    assert sorted(
        re.findall(r"^tenon_apu_g2_host\((\w+) (\S+)\)$", host_cmake, re.M)
    ) == sorted((name, item[0]) for name, item in RUNTIMES.items())
    device_cmake = (TEMPLATE / "device" / "CMakeLists.txt").read_text()
    library = (TEMPLATE / "device" / "gsi_library.h").read_text()
    for name, (_host, task, runtime) in RUNTIMES.items():
        assert task in device_cmake
        assert f"GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE({name}," in library
        assert f'"{name}"' in (ROOT / "allo" / "pim" / runtime).read_text()


def test_typed_and_composed_runtimes_build_only_their_host_target():
    for name in ("tenon_apu_g2_typed", "tenon_apu_g2_composed_dot"):
        runtime = (ROOT / "allo" / "pim" / RUNTIMES[name][2]).read_text()
        assert re.search(r'"--target",\s*"' + name + '"', runtime), name


def test_benchmark_era_constants_are_gone():
    params = (TEMPLATE / "apu_g2_params.h").read_text()
    assert "BMAX" not in params and "U16_GEMM" not in params
    for module, names in (
        (
            apu_g2_costs,
            (
                "APUG2_GEMM_MAX_BATCH_COLUMNS",
                "estimate_apu_g2_u16_gemm_wall_us",
                "estimate_apu_g2_gesummv",
                "estimate_apu_g2_gemv",
                "estimate_apu_g2_atax",
            ),
        ),
        (
            apu_g2_recipe,
            (
                "APUG2_DOT_TILE_MEASURED_POINTS",
                "calibrate_apu_g2_recipe_from_measured_tile",
                "build_apu_g2_u16_dot_tile_recipe",
            ),
        ),
        (
            apu_g2_pipeline,
            (
                "NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION",
                "NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION",
                "NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION",
            ),
        ),
        (pim, ("build_apu_g2_u16_dot_tile_recipe",)),
    ):
        for name in names:
            assert not hasattr(module, name), (module.__name__, name)
    assert apu_g2_costs.apu_g2_cost.materialization_scorer is None
