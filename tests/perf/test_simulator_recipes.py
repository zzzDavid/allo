# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulator build recipes shipped in this repository (release gate G7).

``simulators/`` holds the Tenon-side recipes for the three simulators.  The
Samsung PIMSimulator recipe is complete: a pristine copy of the pinned
upstream sources (``upstream/``, hashed in ``UPSTREAM_MANIFEST.sha256``),
the five patches to upstream files (``patches/``), Tenon's conductor sources
and ``apply.sh``; ``NOTICE.md`` states the terms (release issue P-04).
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test_packaging import sdist_members  # noqa: F401  (pytest fixture)

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "simulators"
PIMSIM = SIM / "PIMSimulator"

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
    "PIMSimulator/NOTICE.md",
    "PIMSimulator/UPSTREAM_MANIFEST.sha256",
    "PIMSimulator/src/pim_driver.cc",
    "PIMSimulator/src/tests/TenonConductor.h",
    "PIMSimulator/src/tests/TenonConductor.inc",
    "PIMSimulator/src/tests/TemporalReduce.inc",
    "PIMSimulator/patches/PIMKernel.h.patch",
    "PIMSimulator/patches/PIMKernel.cpp.patch",
    "PIMSimulator/patches/PIMCmdGen.h.patch",
    "PIMSimulator/patches/PIMCmdGen.cpp.patch",
    "PIMSimulator/patches/Burst.h.patch",
    "PIMSimulator/upstream/LICENSE-PIMSimulator",
    "PIMSimulator/upstream/LICENSE-DRAMSIM2",
    "PIMSimulator/upstream/Sconstruct",
    "PIMSimulator/upstream/src/PIMRank.cpp",
    "PIMSimulator/upstream/src/tests/PIMKernel.cpp",
    "PIMSimulator/upstream/ini/HBM2_samsung_2M_16B_x64.ini",
    "PIMSimulator/upstream/system_hbm_64ch.ini",
)

SAMSUNG_PATCHES = {
    "PIMKernel.h.patch": "src/tests/PIMKernel.h",
    "PIMKernel.cpp.patch": "src/tests/PIMKernel.cpp",
    "PIMCmdGen.h.patch": "src/tests/PIMCmdGen.h",
    "PIMCmdGen.cpp.patch": "src/tests/PIMCmdGen.cpp",
    "Burst.h.patch": "src/Burst.h",
}

OVERLAY_FILES = (
    "Sconstruct",
    "src/pim_driver.cc",
    "src/tests/TenonConductor.h",
    "src/tests/TenonConductor.inc",
    "src/tests/TemporalReduce.inc",
)

UPSTREAM_COMMIT = "3703d1f19c8f027360cc33a3243eb271e3bb6898"
SAMSUNG_NOTICE_PHRASES = (
    "Copyright (C) 2021 Samsung Electronics Co. LTD",
    "express written permission of Samsung Electronics",
    "non-commercial, personal or academic, research purpose",
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
        "PIMSimulator/README.md": UPSTREAM_COMMIT,
        "PIMSimulator/NOTICE.md": UPSTREAM_COMMIT,
        "PIMSimulator/apply.sh": UPSTREAM_COMMIT,
    }
    for relpath, commit in readmes.items():
        assert commit in (SIM / relpath).read_text(), relpath
    upimulator = (SIM / "uPIMulator/README.md").read_text()
    # The upstream MIT notice is reproduced verbatim for the patch.
    assert "Copyright (c) 2024, VIA Research Group at KAIST" in upimulator
    assert "Permission is hereby granted, free of charge" in upimulator


def test_pimsimulator_notice_reproduces_the_samsung_terms():
    """The bundle's NOTICE.md reproduces Samsung's notice verbatim, names what
    is and is not included, and states the redistribution status without
    claiming a permission that is not recorded."""
    notice = (PIMSIM / "NOTICE.md").read_text()
    for phrase in SAMSUNG_NOTICE_PHRASES:
        assert phrase in notice, phrase
    upstream_notice = (PIMSIM / "upstream/LICENSE-PIMSimulator").read_text()
    for phrase in SAMSUNG_NOTICE_PHRASES:
        assert phrase in upstream_notice, phrase
    text = " ".join(notice.split())
    assert "no written permission from Samsung is recorded" in text
    assert "non-commercial" in text
    assert "`data/` and `dump/`" in text  # what is left out, and why
    assert "P-04" in text
    readme = " ".join((PIMSIM / "README.md").read_text().split())
    assert "NOTICE.md" in readme
    assert "python -m allo.pim.samsung_simulator build" in readme
    for patch, upstream in SAMSUNG_PATCHES.items():
        assert f"`{patch}`" in readme and f"`{upstream}`" in readme


def test_samsung_patches_and_upstream_are_tracked_and_shipped():
    """The patches and the upstream copy are tracked (the earlier ignore and
    sdist prune are gone) so that `git archive` and the sdist carry them."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "simulators/PIMSimulator"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0 and tracked.stdout.strip():
        files = set(tracked.stdout.split())
        for patch in SAMSUNG_PATCHES:
            assert f"simulators/PIMSimulator/patches/{patch}" in files, patch
        assert "simulators/PIMSimulator/upstream/LICENSE-PIMSimulator" in files
        assert "simulators/PIMSimulator/UPSTREAM_MANIFEST.sha256" in files
    ignore = (REPO / ".gitignore").read_text()
    assert "simulators/PIMSimulator/patches/" not in ignore
    manifest = (REPO / "MANIFEST.in").read_text().splitlines()
    assert "graft simulators" in manifest
    assert not [line for line in manifest if line.startswith("prune simulators")]


def _manifest() -> dict[str, str]:
    entries = {}
    for line in (PIMSIM / "UPSTREAM_MANIFEST.sha256").read_text().splitlines():
        digest, _, rel = line.partition("  ")
        entries[rel] = digest
    return entries


def test_upstream_manifest_matches_the_bundled_sources():
    """Every bundled upstream file hashes to its manifest entry, nothing is
    missing or unlisted, and the patched files are listed (pristine)."""
    entries = _manifest()
    assert len(entries) >= 60
    upstream = PIMSIM / "upstream"
    present = {
        p.relative_to(upstream).as_posix() for p in upstream.rglob("*") if p.is_file()
    }
    assert present == set(entries), present ^ set(entries)
    for rel, digest in entries.items():
        assert hashlib.sha256((upstream / rel).read_bytes()).hexdigest() == digest, rel
    for upstream_file in SAMSUNG_PATCHES.values():
        assert upstream_file in entries, upstream_file
    for runtime in ("ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini"):
        assert runtime in entries, runtime
    # Not shipped: the benchmark inputs and the upstream build outputs.
    assert not [r for r in entries if r.startswith(("data/", "dump/", "bin/"))]


def _upstream_copy(tmp_path: Path) -> Path:
    tree = tmp_path / "PIMSimulator"
    shutil.copytree(PIMSIM / "upstream", tree)
    return tree


def _run_apply(*args: str, env=None, script: Path = PIMSIM / "apply.sh"):
    environment = dict(os.environ, TENON_SKIP_BUILD="1")
    environment.pop("TENON_PIMSIM_PATCHES", None)
    environment.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        check=False,
    )


def test_pimsimulator_apply_patches_the_bundled_upstream(tmp_path):
    """apply.sh with the bundled patches turns a copy of upstream/ into the
    source tree pim_driver builds from: exactly the five patched files and
    the five Tenon files differ from upstream, the second run is a no-op."""
    if shutil.which("patch") is None:
        pytest.skip("patch(1) is not installed")
    tree = _upstream_copy(tmp_path)
    result = _run_apply(str(tree))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("patched: ") == len(SAMSUNG_PATCHES)
    entries = _manifest()
    changed = {
        rel
        for rel in entries
        if hashlib.sha256((tree / rel).read_bytes()).hexdigest() != entries[rel]
    }
    assert changed == set(SAMSUNG_PATCHES.values()) | {"Sconstruct"}, changed
    for installed in OVERLAY_FILES:
        assert (tree / installed).read_bytes() == (PIMSIM / installed).read_bytes()
    # The patched files include the Tenon conductor and count the vendor
    # generator's invocations.
    assert "tests/TenonConductor.inc" in (tree / "src/tests/PIMKernel.cpp").read_text()
    assert "getInvocationCount" in (tree / "src/tests/PIMCmdGen.h").read_text()
    again = _run_apply(str(tree))
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("already applied: ") == len(SAMSUNG_PATCHES)


def test_pimsimulator_apply_refuses_incomplete_patch_dir(tmp_path):
    tree = _upstream_copy(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "Burst.h.patch").write_text("")
    before = {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()}
    result = _run_apply(str(tree), str(partial))
    assert result.returncode == 2
    assert "missing: PIMKernel.h.patch PIMKernel.cpp.patch" in result.stderr
    assert "Burst.h.patch" not in result.stderr.split("missing:")[1].split(")")[0]
    after = {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()}
    assert after == before


def test_pimsimulator_apply_refuses_a_non_source_tree(tmp_path):
    result = _run_apply(str(tmp_path))
    assert result.returncode == 2
    assert "not a PIMSimulator source tree" in result.stderr


def _synthetic_patches(tmp_path: Path) -> Path:
    """Patches against a dummy checkout (no Samsung content)."""
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


def _dummy_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "PIMSimulator"
    for upstream in SAMSUNG_PATCHES.values():
        (checkout / upstream).parent.mkdir(parents=True, exist_ok=True)
        (checkout / upstream).write_text("// upstream\n")
    return checkout


@pytest.mark.parametrize("route", ["argument", "environment"])
def test_pimsimulator_apply_uses_supplied_patches(tmp_path, route):
    """A user's own patch set still takes precedence over the bundled one."""
    if shutil.which("patch") is None:
        pytest.skip("patch(1) is not installed")
    checkout = _dummy_checkout(tmp_path)
    patches = _synthetic_patches(tmp_path)
    args, env = [str(checkout)], {}
    if route == "argument":
        args.append(str(patches))
    else:
        env["TENON_PIMSIM_PATCHES"] = str(patches)
    result = _run_apply(*args, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("patched: ") == len(SAMSUNG_PATCHES)
    for upstream in SAMSUNG_PATCHES.values():
        assert (checkout / upstream).read_text() == "// upstream\n// tenon\n"


def test_sdist_ships_the_simulator_recipes(sdist_members):  # noqa: F811
    members, _top = sdist_members
    missing = [
        f"simulators/{relpath}"
        for relpath in RECIPE_FILES
        if f"simulators/{relpath}" not in members
    ]
    assert not missing, missing
    prefix = "simulators/PIMSimulator/upstream/"
    shipped = {m[len(prefix) :] for m in members if m.startswith(prefix)}
    assert set(_manifest()) <= shipped, sorted(set(_manifest()) - shipped)
    assert not [m for m in shipped if m.startswith(("data/", "dump/", "bin/"))]
