# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulator build recipes shipped in this repository (release gate G7).

``simulators/`` holds every redistributable Tenon-side recipe file; the
Samsung-derived PIMSimulator patches are deliberately absent (release issue
P-04), and ``PIMSimulator/apply.sh`` must refuse cleanly without them.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test_packaging import sdist_members  # noqa: F401  (pytest fixture)

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "simulators"

RECIPE_FILES = (
    "README.md",
    "aim_simulator/Dockerfile",
    "aim_simulator/README.md",
    "uPIMulator/tenon-upimulator.patch",
    "uPIMulator/apply.sh",
    "uPIMulator/Dockerfile",
    "uPIMulator/README.md",
    "PIMSimulator/Sconstruct",
    "PIMSimulator/apply.sh",
    "PIMSimulator/README.md",
    "PIMSimulator/src/pim_driver.cc",
    "PIMSimulator/src/tests/TenonConductor.h",
    "PIMSimulator/src/tests/TenonConductor.inc",
    "PIMSimulator/src/tests/TemporalReduce.inc",
)

SAMSUNG_PATCHES = {
    "PIMKernel.h.patch": "src/tests/PIMKernel.h",
    "PIMKernel.cpp.patch": "src/tests/PIMKernel.cpp",
    "PIMCmdGen.h.patch": "src/tests/PIMCmdGen.h",
    "PIMCmdGen.cpp.patch": "src/tests/PIMCmdGen.cpp",
    "Burst.h.patch": "src/Burst.h",
}

LIMITATION = (
    "cannot be built from the release repository alone"
)


@pytest.mark.parametrize("relpath", RECIPE_FILES)
def test_recipe_file_present_and_non_empty(relpath):
    path = SIM / relpath
    assert path.is_file(), path
    assert path.stat().st_size > 0, path


def test_apply_scripts_are_executable():
    for script in ("uPIMulator/apply.sh", "PIMSimulator/apply.sh"):
        assert os.access(SIM / script, os.X_OK), script


def test_pinned_upstream_commits_are_recorded():
    readmes = {
        "aim_simulator/README.md": "0f28a07bdb83e42b9305ad3d45410ebd3aa2c091",
        "uPIMulator/README.md": "870d916334e9ff0b190f555f951a9ec3c4257781",
        "PIMSimulator/README.md": "3703d1f19c8f027360cc33a3243eb271e3bb6898",
    }
    for relpath, commit in readmes.items():
        assert commit in (SIM / relpath).read_text(), relpath
    upimulator = (SIM / "uPIMulator/README.md").read_text()
    # The upstream MIT notice is reproduced verbatim for the patch.
    assert "Copyright (c) 2024, VIA Research Group at KAIST" in upimulator
    assert "Permission is hereby granted, free of charge" in upimulator


def test_pimsimulator_readme_states_the_samsung_limitation():
    text = " ".join((SIM / "PIMSimulator/README.md").read_text().split())
    assert LIMITATION in text
    assert "until Samsung authorizes" in text
    assert "experiments/simulators/setup/PIMSimulator/patches/" in text
    for patch, upstream in SAMSUNG_PATCHES.items():
        assert f"`{patch}`" in text and f"`{upstream}`" in text


def test_no_samsung_patch_is_tracked_or_shipped():
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "simulators"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        assert not [p for p in tracked.stdout.split() if p.endswith(".patch")
                    and p.startswith("simulators/PIMSimulator/")]
    assert "simulators/PIMSimulator/patches/" in (REPO / ".gitignore").read_text()
    manifest = (REPO / "MANIFEST.in").read_text().splitlines()
    assert "graft simulators" in manifest
    assert "prune simulators/PIMSimulator/patches" in manifest


def _pimsim_overlay(tmp_path: Path) -> Path:
    """Copy of simulators/PIMSimulator without any sibling patches/ directory."""
    overlay = tmp_path / "overlay"
    shutil.copytree(SIM / "PIMSimulator", overlay, ignore=shutil.ignore_patterns("patches"))
    return overlay


def _dummy_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "PIMSimulator"
    for upstream in SAMSUNG_PATCHES.values():
        (checkout / upstream).parent.mkdir(parents=True, exist_ok=True)
        (checkout / upstream).write_text("// upstream\n")
    return checkout


def _run_apply(overlay: Path, *args: str, env=None):
    environment = dict(os.environ, TENON_SKIP_BUILD="1")
    environment.pop("TENON_PIMSIM_PATCHES", None)
    environment.update(env or {})
    return subprocess.run(
        ["bash", str(overlay / "apply.sh"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        check=False,
    )


def test_pimsimulator_apply_refuses_without_patches(tmp_path):
    overlay = _pimsim_overlay(tmp_path)
    checkout = _dummy_checkout(tmp_path)
    before = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}

    result = _run_apply(overlay, str(checkout))

    assert result.returncode == 2, result
    message = " ".join(result.stderr.split())
    assert "Samsung-derived PIMSimulator patches are not available" in message
    assert "Samsung HBM-PIM simulator route " + LIMITATION in message
    assert "P-04" in message
    for patch in SAMSUNG_PATCHES:
        assert patch in message
    # Nothing was installed or patched.
    after = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    assert after == before


def test_pimsimulator_apply_refuses_incomplete_patch_dir(tmp_path):
    overlay = _pimsim_overlay(tmp_path)
    checkout = _dummy_checkout(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "Burst.h.patch").write_text("")
    result = _run_apply(overlay, str(checkout), str(partial))
    assert result.returncode == 2
    assert "missing: PIMKernel.h.patch PIMKernel.cpp.patch" in result.stderr
    assert "Burst.h.patch" not in result.stderr.split("missing:")[1].split(")")[0]


def _synthetic_patches(tmp_path: Path) -> Path:
    """Patches against the dummy checkout (no Samsung content)."""
    patches = tmp_path / "user-patches"
    patches.mkdir()
    for patch, upstream in SAMSUNG_PATCHES.items():
        diff = difflib.unified_diff(
            ["// upstream\n"],
            ["// upstream\n", "// tenon\n"],
            fromfile=f"a/{upstream}",
            tofile=f"b/{upstream}",
        )
        (patches / patch).write_text("".join(diff))
    return patches


@pytest.mark.parametrize("route", ["argument", "environment", "sibling"])
def test_pimsimulator_apply_uses_supplied_patches(tmp_path, route):
    if shutil.which("patch") is None:
        pytest.skip("patch(1) is not installed")
    overlay = _pimsim_overlay(tmp_path)
    checkout = _dummy_checkout(tmp_path)
    patches = _synthetic_patches(tmp_path)
    args, env = [str(checkout)], {}
    if route == "argument":
        args.append(str(patches))
    elif route == "environment":
        env["TENON_PIMSIM_PATCHES"] = str(patches)
    else:
        shutil.copytree(patches, overlay / "patches")

    result = _run_apply(overlay, *args, env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("patched: ") == len(SAMSUNG_PATCHES)
    for upstream in SAMSUNG_PATCHES.values():
        assert (checkout / upstream).read_text() == "// upstream\n// tenon\n"
    for installed in (
        "Sconstruct",
        "src/pim_driver.cc",
        "src/tests/TenonConductor.h",
        "src/tests/TenonConductor.inc",
        "src/tests/TemporalReduce.inc",
    ):
        assert (checkout / installed).read_bytes() == (overlay / installed).read_bytes()
    # Idempotent: a second run reports the patches as already applied.
    again = _run_apply(overlay, *args, env=env)
    assert again.returncode == 0
    assert again.stdout.count("already applied: ") == len(SAMSUNG_PATCHES)


def test_sdist_ships_the_simulator_recipes(sdist_members):  # noqa: F811
    members, _top = sdist_members
    missing = [f"simulators/{relpath}" for relpath in RECIPE_FILES
               if f"simulators/{relpath}" not in members]
    assert not missing, missing
    assert not [m for m in members if m.startswith("simulators/PIMSimulator/patches")]
